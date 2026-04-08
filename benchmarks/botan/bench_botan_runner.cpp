#include <eosio/vm/backend.hpp>
#include <eosio/vm/utils.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace eosio::vm;

// The backend gives us access to linear memory via the execution context
// WASI functions need to write to wasm linear memory

// Global pointer to linear memory (set after backend init)
static char* g_memory = nullptr;

void write32(int32_t addr, int32_t val) {
    if (g_memory) memcpy(g_memory + addr, &val, 4);
}
void write64(int32_t addr, int64_t val) {
    if (g_memory) memcpy(g_memory + addr, &val, 8);
}

// Exception handling — allocate from wasm heap (bump allocator)
static int32_t g_heap_ptr = 128*1024; // start at 1MB into linear memory
int32_t cxa_begin_catch(int32_t obj) { return obj; }

int32_t cxa_allocate_exception(int32_t size) {
    int32_t ptr = g_heap_ptr;
    g_heap_ptr += (size + 15) & ~15; // align to 16
    return ptr;
}
void cxa_throw(int32_t obj, int32_t type, int32_t dtor) {
    // Wasm expects this to unwind to catch handler — but we can't do real
    // wasm exception handling. Just abort with a message.
    fprintf(stderr, "cxa_throw: wasm threw exception (obj=%d type=%d)\n", obj, type);
    abort();
}

// WASI mocks
int32_t wasi_args_get(int32_t argv, int32_t buf) { return 0; }
int32_t wasi_args_sizes_get(int32_t argc_ptr, int32_t bufsz_ptr) {
    write32(argc_ptr, 0); write32(bufsz_ptr, 0); return 0;
}
int32_t wasi_environ_get(int32_t, int32_t) { return 0; }
int32_t wasi_environ_sizes_get(int32_t count_ptr, int32_t bufsz_ptr) {
    write32(count_ptr, 1); write32(bufsz_ptr, 6); return 0; // 1 env var, "A=B\0" + padding
}
int32_t wasi_environ_get_impl(int32_t argv, int32_t buf) {
    if (g_memory) {
        // Write one env var "A=B\0"
        write32(argv, buf);
        g_memory[buf] = 'A'; g_memory[buf+1] = '='; g_memory[buf+2] = 'B'; g_memory[buf+3] = 0;
    }
    return 0;
}
int32_t wasi_clock_time_get(int32_t id, int64_t prec, int32_t out) {
    write64(out, 0); return 0;
}
int32_t wasi_fd_close(int32_t) { return 0; }
int32_t wasi_fd_fdstat_get(int32_t fd, int32_t buf) {
    // fdstat: filetype(1b) + fdflags(2b) + pad(5b) + rights_base(8b) + rights_inh(8b) = 24 bytes
    if (g_memory) {
        memset(g_memory + buf, 0, 24);
        // filetype: 2 = character_device (for stdin/stdout/stderr)
        if (fd <= 2) g_memory[buf] = 2;
        // rights_base: allow all
        uint64_t rights = ~uint64_t(0);
        memcpy(g_memory + buf + 8, &rights, 8);
        memcpy(g_memory + buf + 16, &rights, 8);
    }
    return 0;
}
int32_t wasi_fd_fdstat_set_flags(int32_t, int32_t) { return 0; }
int32_t wasi_fd_prestat_get(int32_t, int32_t) { return 8; }
int32_t wasi_fd_prestat_dir_name(int32_t, int32_t, int32_t) { return 8; }
int32_t wasi_fd_read(int32_t, int32_t, int32_t, int32_t nread) { write32(nread, 0); return 0; }
int32_t wasi_fd_seek(int32_t, int64_t, int32_t, int32_t out) { write64(out, 0); return 0; }
int32_t wasi_fd_write(int32_t fd, int32_t iovs, int32_t iovs_len, int32_t nwritten) {
    // Sum iov lengths
    int32_t total = 0;
    if (g_memory) {
        for (int32_t i = 0; i < iovs_len; i++) {
            int32_t len;
            memcpy(&len, g_memory + iovs + i*8 + 4, 4);
            total += len;
        }
    }
    write32(nwritten, total);
    return 0;
}
int32_t wasi_path_open(int32_t,int32_t,int32_t,int32_t,int32_t,int64_t,int64_t,int32_t,int32_t) { return 76; } // ENOTCAPABLE
void wasi_proc_exit(int32_t code) {
    // Don't _exit — throw to unwind back to caller
    throw std::runtime_error("proc_exit");
}

using rhf_t = registered_host_functions<standalone_function_t>;

int main() {
    rhf_t::add<&cxa_allocate_exception>("env", "__cxa_allocate_exception");
    rhf_t::add<&cxa_throw>("env", "__cxa_throw");
    rhf_t::add<&cxa_begin_catch>("env", "__cxa_begin_catch");
    rhf_t::add<&wasi_args_get>("wasi_snapshot_preview1", "args_get");
    rhf_t::add<&wasi_args_sizes_get>("wasi_snapshot_preview1", "args_sizes_get");
    rhf_t::add<&wasi_environ_get>("wasi_snapshot_preview1", "environ_get");
    rhf_t::add<&wasi_environ_sizes_get>("wasi_snapshot_preview1", "environ_sizes_get");
    rhf_t::add<&wasi_clock_time_get>("wasi_snapshot_preview1", "clock_time_get");
    rhf_t::add<&wasi_fd_close>("wasi_snapshot_preview1", "fd_close");
    rhf_t::add<&wasi_fd_fdstat_get>("wasi_snapshot_preview1", "fd_fdstat_get");
    rhf_t::add<&wasi_fd_fdstat_set_flags>("wasi_snapshot_preview1", "fd_fdstat_set_flags");
    rhf_t::add<&wasi_fd_prestat_get>("wasi_snapshot_preview1", "fd_prestat_get");
    rhf_t::add<&wasi_fd_prestat_dir_name>("wasi_snapshot_preview1", "fd_prestat_dir_name");
    rhf_t::add<&wasi_fd_read>("wasi_snapshot_preview1", "fd_read");
    rhf_t::add<&wasi_fd_seek>("wasi_snapshot_preview1", "fd_seek");
    rhf_t::add<&wasi_fd_write>("wasi_snapshot_preview1", "fd_write");
    rhf_t::add<&wasi_path_open>("wasi_snapshot_preview1", "path_open");
    rhf_t::add<&wasi_proc_exit>("wasi_snapshot_preview1", "proc_exit");

    auto wasm = read_wasm("botan/bench_botan_verify.wasm");
    printf("Botan ECDSA verify: %.1f MB wasm\n", wasm.size()/1024.0/1024.0);

    wasm_allocator wa;
    auto t1 = std::chrono::high_resolution_clock::now();
    backend<rhf_t, jit2> bkend(wasm, &wa);
    rhf_t::resolve(bkend.get_module());
    bkend.initialize(nullptr);
    auto t2 = std::chrono::high_resolution_clock::now();
    printf("JIT2 compile: %.0f ms\n", std::chrono::duration<double, std::milli>(t2-t1).count());

    // Set linear memory pointer for WASI mocks — must be before _start
    g_memory = bkend.get_context().linear_memory();
    printf("Linear memory: %p\n", (void*)g_memory);

    // Initialize WASI + C++ constructors (Botan EC group tables)
    printf("Running _initialize...\n"); fflush(stdout);
    try {
        bkend.call_with_return("env", "_initialize");
        printf("_initialize OK\n");
    } catch (const std::exception& e) {
        printf("_initialize failed: %s\n", e.what());
        fflush(stdout); _exit(1);
    }

    // Warmup
    printf("Running warmup...\n"); fflush(stdout);
    try {
        auto r = bkend.call_with_return("env", "bench_verify", (int32_t)1);
        printf("warmup: %ld\n", r ? r->to_i64() : -1);
    } catch (const std::exception& e) {
        printf("warmup failed: %s\n", e.what());
        fflush(stdout); _exit(1);
    }

    // Benchmark
    printf("Benchmarking 100 verifies...\n"); fflush(stdout);
    auto t3 = std::chrono::high_resolution_clock::now();
    auto r = bkend.call_with_return("env", "bench_verify", (int32_t)100);
    auto t4 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t4-t3).count();
    printf("\n100 ECDSA verifies (Botan secp256k1):\n");
    printf("  JIT2: %.1f ms total, %.2f ms/verify\n", ms, ms/100);
    printf("  result: %ld (expect 100)\n", r ? r->to_i64() : -1);

    fflush(stdout);
    _exit(0);
}
