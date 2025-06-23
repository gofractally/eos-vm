#pragma once

/*
 * definitions from https://github.com/WebAssembly/design/blob/master/BinaryEncoding.md
 */

#include <eosio/vm/allocator.hpp>
#include <eosio/vm/guarded_ptr.hpp>
#include <eosio/vm/opcodes.hpp>
#include <eosio/vm/vector.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

namespace eosio { namespace vm {
   enum types { i32 = 0x7f, i64 = 0x7e, f32 = 0x7d, f64 = 0x7c, v128 = 0x7b, anyfunc = 0x70, funcref = anyfunc, func = 0x60, pseudo = 0x40, ret_void };

   enum external_kind { Function = 0, Table = 1, Memory = 2, Global = 3 };

   typedef uint8_t value_type;
   typedef uint8_t block_type;
   typedef uint8_t elem_type;

   template <typename T>
   using guarded_vector = managed_vector<T, growable_allocator>;

   struct activation_frame {
      opcode* pc;
      uint32_t last_op_index;
   };

   struct resizable_limits {
      bool     flags;
      uint32_t initial;
      uint32_t maximum = 0;
   };

   struct func_type {
      value_type                 form; // value for the func type constructor
      std::vector<value_type>    param_types;
      uint8_t                    return_count;
      value_type                 return_type;
   };

   inline bool operator==(const func_type& lhs, const func_type& rhs) {
      return lhs.form == rhs.form &&
        lhs.param_types.size() == rhs.param_types.size() &&
        std::equal(lhs.param_types.data(), lhs.param_types.data() + lhs.param_types.size(), rhs.param_types.data()) &&
        lhs.return_count == rhs.return_count &&
        (lhs.return_count || lhs.return_type == rhs.return_type);
   }

   union expr_value {
      int32_t  i32;
      int64_t  i64;
      uint32_t f32;
      uint64_t f64;
      v128_t   v128;
   };

   struct init_expr {
      uint8_t    opcode;
      expr_value value;
   };

   struct global_type {
      value_type content_type;
      bool       mutability;
   };

   struct global_variable {
      global_type type;
      init_expr   init;
   };

   struct table_type {
      elem_type        element_type;
      resizable_limits limits;
   };

   struct table_entry {
      std::uint32_t type;
      std::uint32_t index;
      // The code writer is responsible for filling this field
      const void*   code_ptr;
   };

   struct memory_type {
      resizable_limits limits;
   };

   union import_type {
      import_type() {}
      uint32_t    func_t;
      table_type  table_t;
      memory_type mem_t;
      global_type global_t;
   };

   struct import_entry {
      std::vector<uint8_t>    module_str;
      std::vector<uint8_t>    field_str;
      external_kind           kind;
      import_type             type;
   };

   struct export_entry {
      std::vector<uint8_t>    field_str;
      external_kind           kind;
      uint32_t                index;
   };

   enum class elem_mode { active, passive, declarative };

   struct elem_segment {
      uint32_t                    index;
      init_expr                   offset;
      elem_mode                   mode;
      std::vector<table_entry>    elems;
   };

   struct local_entry {
      uint32_t   count;
      value_type type;
   };

   union native_value {
      native_value() = default;
      constexpr native_value(uint32_t arg) : i32(arg) {}
      constexpr native_value(uint64_t arg) : i64(arg) {}
      constexpr native_value(float arg) : f32(arg) {}
      constexpr native_value(double arg) : f64(arg) {}
      uint32_t i32;
      uint64_t i64;
      float f32;
      double f64;
   };

   union native_value_extended {
      native_value scalar;
      v128_t vector;
   };

   struct function_body {
      uint32_t                    size;
      std::vector<local_entry>    locals;
      opcode*                     code;
      std::size_t                 jit_code_offset;
      std::uint32_t               stack_size = 0;
   };

   struct data_segment {
      uint32_t                index;
      init_expr               offset;
      bool                    passive;
      std::vector<uint8_t>    data;
   };

   using wasm_code     = std::vector<uint8_t>;
   using wasm_code_ptr = guarded_ptr<uint8_t>;
   typedef std::uint32_t  wasm_ptr_t;
   typedef std::uint32_t  wasm_size_t;

   struct name_assoc {
      std::uint32_t idx;
      std::vector<uint8_t> name;
   };
   struct indirect_name_assoc {
      std::uint32_t idx;
      std::vector<name_assoc> namemap;
   };
   struct name_section {
      std::optional<std::vector<uint8_t>> module_name;
      std::optional<std::vector<name_assoc>> function_names;
      std::optional<std::vector<indirect_name_assoc>> local_names;
   };

   struct module {
      growable_allocator              allocator;
      uint32_t                        start     = std::numeric_limits<uint32_t>::max();
      std::vector<func_type>          types;
      std::vector<import_entry>       imports;
      std::vector<uint32_t>           functions;
      std::vector<table_type>         tables;
      std::vector<memory_type>        memories;
      std::vector<global_variable>    globals;
      std::vector<export_entry>       exports;
      std::vector<elem_segment>       elements;
      std::vector<function_body>      code;
      std::vector<data_segment>       data;

      // Custom sections:
      std::optional<name_section> names;

      // not part of the spec for WASM
      std::vector<uint32_t>    import_functions;
      uint64_t                 maximum_stack    = 0;
      // The stack limit can be tracked as either frames or bytes
      bool                     stack_limit_is_bytes = false;
      // Emulate eosio bugs
      bool                     eosio_fp             = false;
      // If non-null, indicates that the parser encountered an error
      // that would prevent successful instantiation.  Must refer
      // to memory with static storage duration.
      const char *             error            = nullptr;

      void finalize() {
         import_functions.resize(get_imported_functions_size());
         allocator.finalize();
      }
      uint32_t get_imported_functions_size() const {
         return get_imported_functions_size_impl(imports);
      }
      template<typename Imports>
      static uint32_t get_imported_functions_size_impl(const Imports& imports) {
         uint32_t number_of_imports = 0;
         const auto sz = imports.size();
         // we don't want to use `imports[i]` or `imports.at(i)` since these do an unnecessary check
         // `EOS_VM_ASSERT(i < _size)`. The check is unnecessary since we iterate from `0` to `_size`.
         // So get the pointer to the first element and dereference it directly.
         // ------------------------------------------------------------------------------------------------
         const auto data = imports.data();
         for (uint32_t i = 0; i < sz; i++) {
            if (data[i].kind == external_kind::Function)
               number_of_imports++;
         }
         return number_of_imports;
      }
      inline uint32_t get_functions_size() const { return functions.size(); }
      inline uint32_t get_functions_total() const { return get_imported_functions_size() + get_functions_size(); }
      inline opcode* get_function_pc( uint32_t fidx ) const {
         EOS_VM_ASSERT( fidx >= get_imported_functions_size(), wasm_interpreter_exception, "trying to get the PC of an imported function" );
         return code.at(fidx-get_imported_functions_size()).code;
      }

      inline uint32_t get_function_locals_size(uint32_t index) const {
         EOS_VM_ASSERT(index >= get_imported_functions_size(), wasm_interpreter_exception, "imported functions do not have locals");
         return code.at(index - get_imported_functions_size()).locals.size();
      }

      auto& get_function_type(uint32_t index) const {
         if (index < get_imported_functions_size())
            return types[imports[index].type.func_t];
         return types.at(functions.at(index - get_imported_functions_size()));
      }

      uint32_t get_function_stack_size(uint32_t index) const {
         if (!stack_limit_is_bytes) {
            return 1;
         } else if (index < get_imported_functions_size()) {
            return 0;
         } else {
            return code.at(index - get_imported_functions_size()).stack_size;
         }
      }

      uint32_t get_exported_function(const std::string_view str) {
         return get_exported_function_impl(exports, str);
      }

      template<typename Exports>
      static uint32_t get_exported_function_impl(const Exports& exports, const std::string_view str) {
         uint32_t index = std::numeric_limits<uint32_t>::max();
         for (uint32_t i = 0; i < exports.size(); i++) {
            if (exports[i].kind == external_kind::Function && exports[i].field_str.size() == str.size() &&
                memcmp((const char*)str.data(), (const char*)exports[i].field_str.data(), exports[i].field_str.size()) ==
                      0) {
               index = exports[i].index;
               break;
            }
         }
         return index;
      }

      static std::size_t get_global_offset(std::uint32_t idx)
      {
         return idx * sizeof(init_expr) + offsetof(init_expr, value);
      }

      bool indirect_table(std::size_t i)
      {
         return indirect_table_impl(*this, i);
      }

      static bool indirect_table_impl(auto& mod, std::size_t i) {
         return i < mod.tables.size() && (mod.tables[i].limits.initial * sizeof(table_entry) > (wasm_allocator::table_size()) - sizeof(void*));
      }
   };
}} // namespace eosio::vm
