#pragma once

#include <eosio/vm/allocator.hpp>
#include <eosio/vm/constants.hpp>
#include <eosio/vm/exceptions.hpp>
#include <eosio/vm/leb128.hpp>
#include <eosio/vm/options.hpp>
#include <eosio/vm/sections.hpp>
#include <eosio/vm/types.hpp>
#include <eosio/vm/utils.hpp>
#include <eosio/vm/vector.hpp>
#include <eosio/vm/debug_info.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

namespace eosio { namespace vm {

   namespace detail {

   static constexpr unsigned get_size_for_type(uint8_t type) {
      switch(type) {
       case types::i32:
       case types::f32:
          return 4;
       case types::i64:
       case types::f64:
          return 8;
       case types::v128:
          return 16;
       default: return 0;
      }
   }

   template<typename Options, typename Enable = void>
   struct max_mutable_globals_checker {
      constexpr void on_mutable_global(const Options&, uint8_t) {}
   };

   template<typename Options>
   using max_mutable_globals_t = decltype(std::declval<Options>().max_mutable_global_bytes);

   template<typename Options>
   struct max_mutable_globals_checker<Options, std::void_t<max_mutable_globals_t<Options>>> {
      static_assert(std::is_unsigned_v<std::decay_t<max_mutable_globals_t<Options>>>, "max_mutable_globals must be an unsigned integer type");
      void on_mutable_global(const Options& options, uint8_t type) {
         unsigned size = get_size_for_type(type);
         _counter += size;
         EOS_VM_ASSERT(_counter <= options.max_mutable_global_bytes && _counter >= size, wasm_parse_exception, "mutable globals exceeded limit");
      }
      std::decay_t<max_mutable_globals_t<Options>> _counter = 0;
   };

#define PARSER_OPTION(name, default_, type)                                   \
   template<typename Options>                                                 \
   type get_ ## name(const Options& options, long) { (void)options; return default_; } \
   template<typename Options>                                                 \
   auto get_ ## name(const Options& options, int) -> decltype(options.name) { \
     return options.name;                                                     \
   }                                                                          \
   template<typename Options>                                                 \
   type get_ ## name(const Options& options) { return detail::get_ ## name(options, 0); }

#define MAX_ELEMENTS(name, default_)\
   PARSER_OPTION(name, default_, std::uint32_t)

   MAX_ELEMENTS(max_table_elements, 0xFFFFFFFFu)
   MAX_ELEMENTS(max_section_elements, 0xFFFFFFFFu)

   MAX_ELEMENTS(max_type_section_elements, detail::get_max_section_elements(options))
   MAX_ELEMENTS(max_import_section_elements, detail::get_max_section_elements(options))
   MAX_ELEMENTS(max_function_section_elements, detail::get_max_section_elements(options))
   MAX_ELEMENTS(max_global_section_elements, detail::get_max_section_elements(options))
   MAX_ELEMENTS(max_export_section_elements, detail::get_max_section_elements(options))
   MAX_ELEMENTS(max_element_section_elements, detail::get_max_section_elements(options))
   MAX_ELEMENTS(max_data_section_elements, detail::get_max_section_elements(options))

   MAX_ELEMENTS(max_element_segment_elements, 0xFFFFFFFFu)
   MAX_ELEMENTS(max_data_segment_bytes, 0xFFFFFFFFu)

   PARSER_OPTION(max_linear_memory_init, 0xFFFFFFFFFFFFFFFFu, std::uint64_t)
   PARSER_OPTION(max_func_local_bytes_flags, max_func_local_bytes_flags_t::locals | max_func_local_bytes_flags_t::stack, max_func_local_bytes_flags_t);

   template<typename Options, typename Enable = void>
   struct max_func_local_bytes_checker {
      explicit max_func_local_bytes_checker(const Options&, const func_type& /*ft*/) {}
      void on_local(const Options&, std::uint8_t, const std::uint32_t) {}
      void push_stack(const Options& /*options*/, std::uint8_t /*type*/) {}
      void pop_stack(std::uint8_t /*type*/) {}
      void push_unreachable() {}
      void pop_unreachable() {}
      static constexpr bool is_defined = false;
   };
   template<typename Options>
   struct max_func_local_bytes_checker<Options, std::void_t<decltype(std::declval<Options>().max_func_local_bytes)>> {
      explicit max_func_local_bytes_checker(const Options& options, const func_type& ft) {
         if ((detail::get_max_func_local_bytes_flags(options) & max_func_local_bytes_flags_t::params) != (max_func_local_bytes_flags_t)0) {
            for(std::uint32_t i = 0; i < ft.param_types.size(); ++i) {
               on_type(options, ft.param_types.at(i));
            }
         }
      }
      void on_type(const Options& options, std::uint8_t type) {
         unsigned size = get_size_for_type(type);
         _count += size;
         EOS_VM_ASSERT(_count <= options.max_func_local_bytes && _count >= size, wasm_parse_exception, "local variable limit exceeded");
      }
      void on_local(const Options& options, std::uint8_t type, std::uint32_t count) {
         if ((detail::get_max_func_local_bytes_flags(options) & max_func_local_bytes_flags_t::locals) != (max_func_local_bytes_flags_t)0) {
            uint64_t size = get_size_for_type(type);
            size *= count;
            _count += size;
            EOS_VM_ASSERT(_count <= options.max_func_local_bytes && _count >= size, wasm_parse_exception, "local variable limit exceeded");
         }
      }
      std::decay_t<decltype(std::declval<Options>().max_func_local_bytes)> _count = 0;
      static constexpr bool is_defined = true;
   };
   template<typename Options>
   constexpr auto get_max_func_local_bytes_no_stack_c(int) -> std::enable_if_t<std::is_pointer_v<decltype(&Options::max_func_local_bytes_flags)>, bool>
   { return (Options::max_func_local_bytes_flags & max_func_local_bytes_flags_t::stack) == (max_func_local_bytes_flags_t)0; }
   template<typename Options>
   constexpr auto get_max_func_local_bytes_no_stack_c(long) -> bool { return false; }

   template<typename Options, typename Enable = void>
   struct max_func_local_bytes_stack_checker : max_func_local_bytes_checker<Options> {
      explicit constexpr max_func_local_bytes_stack_checker(const max_func_local_bytes_checker<Options>& base) : max_func_local_bytes_checker<Options>(base) {}
      void push_stack(const Options& options, std::uint8_t type) {
         if(unreachable_depth == 0 && (detail::get_max_func_local_bytes_flags(options) & max_func_local_bytes_flags_t::stack) != (max_func_local_bytes_flags_t)0) {
            this->on_type(options, type);
         }
      }
      void pop_stack(const Options& options, std::uint8_t type) {
         if(unreachable_depth == 0 && (detail::get_max_func_local_bytes_flags(options) & max_func_local_bytes_flags_t::stack) != (max_func_local_bytes_flags_t)0) {
            this->_count -= get_size_for_type(type);
         }
      }
      void push_unreachable() {
         ++unreachable_depth;
      }
      void pop_unreachable() {
         --unreachable_depth;
      }
      std::uint32_t unreachable_depth = 0;
   };
   template<typename Options>
   struct max_func_local_bytes_stack_checker<Options, std::enable_if_t<!max_func_local_bytes_checker<Options>::is_defined ||
                                                                       get_max_func_local_bytes_no_stack_c<Options>(0)>> {
      explicit constexpr max_func_local_bytes_stack_checker(const max_func_local_bytes_checker<Options>&) {}
      void push_stack(const Options& /*options*/, std::uint8_t /*type*/) {}
      void pop_stack(const Options& /*options*/, std::uint8_t /*type*/) {}
      void push_unreachable() {}
      void pop_unreachable() {}
   };

   MAX_ELEMENTS(max_local_sets, 0xFFFFFFFFu)
   MAX_ELEMENTS(max_nested_structures, 0xFFFFFFFFu)
   MAX_ELEMENTS(max_br_table_elements, 0xFFFFFFFFu)

   // Matches the behavior of eosio::chain::wasm_validations::nested_validator
   template<typename Options, typename Enable = void>
   struct eosio_max_nested_structures_checker {
      void on_control(const Options&) {}
      void on_end(const Options&) {}
   };
   template<typename Options>
   struct eosio_max_nested_structures_checker<Options, std::void_t<decltype(std::declval<Options>().eosio_max_nested_structures)>> {
      void on_control(const Options& options) {
         ++_count;
         EOS_VM_ASSERT(_count <= options.eosio_max_nested_structures, wasm_parse_exception, "Nested depth exceeded");
      }
      void on_end(const Options& options) {
         if(_count == 0) ++_count;
         else --_count;
      }
      std::decay_t<decltype(std::declval<Options>().eosio_max_nested_structures)> _count = 0;
   };

   MAX_ELEMENTS(max_symbol_bytes, 0xFFFFFFFFu)
   MAX_ELEMENTS(max_memory_offset, 0xFFFFFFFFu)
   MAX_ELEMENTS(max_code_bytes, 0xFFFFFFFFu)
   MAX_ELEMENTS(max_pages, 0xFFFFFFFFu)
   MAX_ELEMENTS(max_call_depth, 251)

   PARSER_OPTION(forbid_export_mutable_globals, false, bool);
   PARSER_OPTION(allow_code_after_function_end, false, bool);
   PARSER_OPTION(allow_u32_limits_flags, false, bool);
   PARSER_OPTION(allow_invalid_empty_local_set, false, bool);
   PARSER_OPTION(eosio_fp, false, bool);

   PARSER_OPTION(allow_zero_blocktype, false, bool)

   PARSER_OPTION(parse_custom_section_name, false, bool);

   PARSER_OPTION(enable_simd, true, bool)
   PARSER_OPTION(enable_bulk_memory, true, bool)
   PARSER_OPTION(enable_sign_ext, true, bool)
   PARSER_OPTION(enable_nontrapping_fptoint, true, bool)

#undef MAX_ELEMENTS
#undef PARSER_OPTION

   template <typename Options>
   concept has_max_stack_bytes = requires(const Options& o) {
       o.max_stack_bytes;
   };

   // Returns either max_call_depth or max_stack_bytes, depending on which
   // on was enabled.
   template <typename Options>
   constexpr std::uint32_t choose_stack_limit(const Options& options)
   {
      if constexpr (has_max_stack_bytes<Options>)
      {
         static_assert(!requires{options.max_call_depth;}, "max_call_depth and max_stack_bytes are incompatible");
         return options.max_stack_bytes;
      }
      else
      {
         return get_max_call_depth(options);
      }
   }

   // max_stack_bytes depends on max_func_local_bytes and uses the
   // same accounting.
   template<typename Options>
   struct max_stack_usage_calculator
   {
      void update(const max_func_local_bytes_stack_checker<Options>&) {}
   };
   template<has_max_stack_bytes Options>
   struct max_stack_usage_calculator<Options>
   {
      void update(const max_func_local_bytes_stack_checker<Options>& x)
      {
         _max = std::max(x._count, _max);
      }
      std::decay_t<decltype(std::declval<Options>().max_stack_bytes)> _max = 0;
   };

   }

   template <typename Writer, typename Options = default_options, typename DebugInfo = null_debug_info>
   class binary_parser {
    public:
      explicit binary_parser(growable_allocator& alloc, const Options& options = Options{}) : _allocator(alloc), _options(options) {}

      template <typename T>
      using vec = guarded_vector<T>;

      static inline uint8_t parse_varuint1(wasm_code_ptr& code) { return varuint<1>(code).to(); }

      static inline uint8_t parse_varuint7(wasm_code_ptr& code) { return varuint<7>(code).to(); }

      static inline uint32_t parse_varuint32(wasm_code_ptr& code) { return varuint<32>(code).to(); }

      static inline int8_t parse_varint7(wasm_code_ptr& code) { return varint<7>(code).to(); }

      static inline int32_t parse_varint32(wasm_code_ptr& code) { return varint<32>(code).to(); }

      static inline int64_t parse_varint64(wasm_code_ptr& code) { return varint<64>(code).to(); }

      int validate_utf8_code_point(wasm_code_ptr& code) {
         unsigned char ch = *code++;
         if (ch < 0x80) {
            return 1;
         } else if(ch < 0xE0) {
           EOS_VM_ASSERT((ch & 0xC0) == 0xC0, wasm_parse_exception, "invalid utf8 encoding");
            unsigned char b2 = *code++;
            EOS_VM_ASSERT((b2 & 0xC0) == 0x80, wasm_parse_exception, "invalid utf8 encoding");
            uint32_t code_point =
              (static_cast<uint32_t>(ch - 0xC0u) << 6u) +
              (static_cast<uint32_t>(b2 - 0x80u));
            EOS_VM_ASSERT(0x80 <= code_point && code_point < 0x800, wasm_parse_exception, "invalid utf8 encoding");
            return 2;
         } else if(ch < 0xF0) {
            unsigned char b2 = *code++;
            EOS_VM_ASSERT((b2 & 0xC0) == 0x80, wasm_parse_exception, "invalid utf8 encoding");
            unsigned char b3 = *code++;
            EOS_VM_ASSERT((b3 & 0xC0) == 0x80, wasm_parse_exception, "invalid utf8 encoding");
            uint32_t code_point =
              (static_cast<uint32_t>(ch - 0xE0u) << 12u) +
              (static_cast<uint32_t>(b2 - 0x80u) << 6u) +
              (static_cast<uint32_t>(b3 - 0x80u));
            EOS_VM_ASSERT((0x800 <= code_point && code_point < 0xD800) ||
                          (0xE000 <= code_point && code_point < 0x10000),
                          wasm_parse_exception, "invalid utf8 encoding");
            return 3;
         } else if (ch < 0xF8) {
            unsigned char b2 = *code++;
            EOS_VM_ASSERT((b2 & 0xC0) == 0x80, wasm_parse_exception, "invalid utf8 encoding");
            unsigned char b3 = *code++;
            EOS_VM_ASSERT((b3 & 0xC0) == 0x80, wasm_parse_exception, "invalid utf8 encoding");
            unsigned char b4 = *code++;
            EOS_VM_ASSERT((b4 & 0xC0) == 0x80, wasm_parse_exception, "invalid utf8 encoding");
            uint32_t code_point =
              (static_cast<uint32_t>(ch - 0xF0u) << 18u) +
              (static_cast<uint32_t>(b2 - 0x80u) << 12u) +
              (static_cast<uint32_t>(b3 - 0x80u) << 6u) +
              (static_cast<uint32_t>(b4 - 0x80u));
            EOS_VM_ASSERT((0x10000 <= code_point && code_point < 0x110000),
                          wasm_parse_exception, "invalid utf8 encoding");
            return 4;
         }
         EOS_VM_ASSERT(false, wasm_parse_exception, "invalid utf8 encoding");
      }

      void validate_utf8_string(wasm_code_ptr& code, uint32_t bytes) {
         while(bytes != 0) {
            bytes -= validate_utf8_code_point(code);
         }
      }

      std::vector<uint8_t> parse_utf8_string(wasm_code_ptr& code, std::uint32_t max_size) {
         auto len        = parse_varuint32(code);
         EOS_VM_ASSERT(len <= max_size, wasm_parse_exception, "name too long");
         auto guard = code.scoped_shrink_bounds(len);
         auto result = std::vector<uint8_t>(code.raw(), code.raw() + len);
         validate_utf8_string(code, len);
         return result;
      }

      template<typename T>
      T parse_raw(wasm_code_ptr& code) {
         static_assert(std::is_arithmetic_v<T>, "Can only read builtin types");
         auto guard = code.scoped_shrink_bounds(sizeof(T));
         T result;
         memcpy(&result, code.raw(), sizeof(T));
         code += sizeof(T);
         return result;
      }

      v128_t parse_v128(wasm_code_ptr& code) {
         static_assert(sizeof(v128_t) == 16, "sanity check for layout");
         auto guard = code.scoped_shrink_bounds(sizeof(v128_t));
         v128_t result;
         memcpy(&result, code.raw(), sizeof(v128_t));
         code += sizeof(v128_t);
         return result;
      }

      inline module& parse_module(wasm_code& code, module& mod, DebugInfo& debug) {
         wasm_code_ptr cp(code.data(), code.size());
         parse_module(cp, code.size(), mod, debug);
         return mod;
      }

      inline module& parse_module2(wasm_code_ptr& code_ptr, size_t sz, module& mod, DebugInfo& debug) {
         parse_module(code_ptr, sz, mod, debug);
         return mod;
      }

      static constexpr auto make_section_order() {
         std::array<std::uint8_t, section_id::num_of_elems> result{};
         std::uint8_t i = 1;
         for (std::uint8_t sec : {type_section, import_section, function_section, table_section, memory_section, global_section, export_section, start_section, element_section, data_count_section, code_section, data_section}) {
            result[sec] = i++;
         }
         return result;
      }

      void parse_module(wasm_code_ptr& code_ptr, size_t sz, module& mod, DebugInfo& debug) {
         _mod = &mod;
         _mod->eosio_fp = detail::get_eosio_fp(_options);
         EOS_VM_ASSERT(parse_magic(code_ptr) == constants::magic, wasm_parse_exception, "magic number did not match");
         EOS_VM_ASSERT(parse_version(code_ptr) == constants::version, wasm_parse_exception,
                       "version number did not match");
         uint8_t highest_section_id = 0;
         constexpr auto order = make_section_order();
         for (;;) {
            if (code_ptr.offset() == sz)
               break;
            auto id = parse_section_id(code_ptr);
            auto len = parse_section_payload_len(code_ptr);

            EOS_VM_ASSERT(id < order.size(), wasm_parse_exception, "invalid section id");
            EOS_VM_ASSERT(id == 0 || order[id] > highest_section_id, wasm_parse_exception, "section out of order");
            highest_section_id = std::max(highest_section_id, order[id]);

            auto section_guard = code_ptr.scoped_consume_items(len);

            switch (id) {
               case section_id::custom_section: parse_custom(code_ptr); break;
               case section_id::type_section: parse_section<section_id::type_section>(code_ptr, mod.types); break;
               case section_id::import_section: parse_section<section_id::import_section>(code_ptr, mod.imports); break;
               case section_id::function_section:
                  parse_section<section_id::function_section>(code_ptr, mod.functions);
                  normalize_types();
                  break;
               case section_id::table_section: parse_section<section_id::table_section>(code_ptr, mod.tables); break;
               case section_id::memory_section:
                  parse_section<section_id::memory_section>(code_ptr, mod.memories);
                  break;
               case section_id::global_section: parse_section<section_id::global_section>(code_ptr, mod.globals); break;
               case section_id::export_section:
                  parse_section<section_id::export_section>(code_ptr, mod.exports);
                  validate_exports();
                  break;
               case section_id::start_section: parse_section<section_id::start_section>(code_ptr, mod.start); break;
               case section_id::element_section:
                  parse_section<section_id::element_section>(code_ptr, mod.elements);
                  break;
               case section_id::code_section: parse_section<section_id::code_section>(code_ptr, mod.code); break;
               case section_id::data_section: parse_section<section_id::data_section>(code_ptr, mod.data); break;
               case section_id::data_count_section: parse_section<section_id::data_count_section>(code_ptr, _datacount); break;
               default: EOS_VM_ASSERT(false, wasm_parse_exception, "error invalid section id");
            }
         }
         EOS_VM_ASSERT(_mod->code.size() == _mod->functions.size(), wasm_parse_exception, "code section must have the same size as the function section" );

         _mod->stack_limit_is_bytes = detail::has_max_stack_bytes<Options>;
         debug.set(std::move(imap));
         debug.relocate(_allocator.get_code_start());
      }

      inline uint32_t parse_magic(wasm_code_ptr& code) {
         return parse_raw<uint32_t>(code);
      }
      inline uint32_t parse_version(wasm_code_ptr& code) {
         return parse_raw<uint32_t>(code);
      }
      inline uint8_t  parse_section_id(wasm_code_ptr& code) { return *code++; }
      inline uint32_t parse_section_payload_len(wasm_code_ptr& code) {
         return parse_varuint32(code);
      }

      inline void parse_custom(wasm_code_ptr& code) {
         auto section_name = parse_utf8_string(code, 0xFFFFFFFFu); // ignored, but needs to be validated
         if(detail::get_parse_custom_section_name(_options) &&
            section_name.size() == 4 && std::memcmp(section_name.data(), "name", 4) == 0) {
            parse_name_section(code);
         } else {
            // skip to the end of the section
            code += code.bounds() - code.offset();
         }
      }

      inline void parse_name_map(wasm_code_ptr& code, std::vector<name_assoc>& map) {
        for(uint32_t i = 0; i < map.size(); ++i) {
            map[i].idx = parse_varuint32(code);
            map[i].name = parse_utf8_string(code, 0xFFFFFFFFu);
         }
      }

      inline void parse_name_section(wasm_code_ptr& code) {
         _mod->names.emplace();
         if(code.bounds() == code.offset()) return;
         if(*code == 0) {
            ++code;
            auto subsection_guard = code.scoped_consume_items(parse_varuint32(code));
            _mod->names->module_name.emplace(parse_utf8_string(code, 0xFFFFFFFFu));
         }
         if(code.bounds() == code.offset()) return;
         if(*code == 1) {
            ++code;
            auto subsection_guard = code.scoped_consume_items(parse_varuint32(code));
            uint32_t size = parse_varuint32(code);
            _mod->names->function_names.emplace(size);
            parse_name_map(code, *_mod->names->function_names);
         }
         if(code.bounds() == code.offset()) return;
         if(*code == 2) {
            ++code;
            auto subsection_guard = code.scoped_consume_items(parse_varuint32(code));
            uint32_t size = parse_varuint32(code);
            _mod->names->local_names.emplace(size);
            for(uint32_t i = 0; i < size; ++i) {
               auto& [idx,namemap] = (*_mod->names->local_names)[i];
               idx = parse_varuint32(code);
               uint32_t local_size = parse_varuint32(code);
               namemap = std::vector<name_assoc>(local_size);
               parse_name_map(code, namemap);
            }
         }
         if(code.bounds() == code.offset()) return;
         EOS_VM_ASSERT(false, wasm_parse_exception, "Invalid subsection Id");
      }

      void parse_import_entry(wasm_code_ptr& code, import_entry& entry) {
         entry.module_str = parse_utf8_string(code, detail::get_max_symbol_bytes(_options));
         entry.field_str = parse_utf8_string(code, detail::get_max_symbol_bytes(_options));
         entry.kind = (external_kind)(*code++);
         auto type  = parse_varuint32(code);
         switch ((uint8_t)entry.kind) {
            case external_kind::Function:
               entry.type.func_t = type;
               EOS_VM_ASSERT(type < _mod->types.size(), wasm_parse_exception, "Invalid function type");
               break;
            default: EOS_VM_ASSERT(false, wasm_unsupported_import_exception, "only function imports are supported");
         }
      }

      uint8_t parse_flags(wasm_code_ptr& code) {
         if (detail::get_allow_u32_limits_flags(_options)) {
            return parse_varuint32(code) & 0x1;
         } else {
            EOS_VM_ASSERT(*code == 0x0 || *code == 0x1, wasm_parse_exception, "invalid flags");
            return *code++;
         }
      }

      void parse_table_type(wasm_code_ptr& code, table_type& tt) {
         tt.element_type   = *code++;
         EOS_VM_ASSERT(tt.element_type == types::anyfunc, wasm_parse_exception, "table must have type anyfunc");
         tt.limits.flags   = parse_flags(code);
         tt.limits.initial = parse_varuint32(code);
         if (tt.limits.flags) {
            tt.limits.maximum = parse_varuint32(code);
            EOS_VM_ASSERT(tt.limits.initial <= tt.limits.maximum, wasm_parse_exception, "table max size less than min size");
         }
         EOS_VM_ASSERT(tt.limits.initial <= detail::get_max_table_elements(_options), wasm_parse_exception, "table size exceeds limit");
      }

      void parse_global_variable(wasm_code_ptr& code, global_variable& gv) {
         uint8_t ct           = *code++;
         gv.type.content_type = ct;
         EOS_VM_ASSERT(ct == types::i32 || ct == types::i64 || ct == types::f32 || ct == types::f64 || (ct == types::v128 && detail::get_enable_simd(_options)),
                       wasm_parse_exception, "invalid global content type");

         gv.type.mutability = parse_varuint1(code);
         if(gv.type.mutability)
            on_mutable_global(ct);
         parse_init_expr(code, gv.init, ct);
      }

      void parse_memory_type(wasm_code_ptr& code, memory_type& mt) {
         mt.limits.flags   = parse_flags(code);
         mt.limits.initial = parse_varuint32(code);
         // Implementation limits
         EOS_VM_ASSERT(mt.limits.initial <= detail::get_max_pages(_options), wasm_parse_exception, "initial memory out of range");
         // WASM specification
         EOS_VM_ASSERT(mt.limits.initial <= 65536u, wasm_parse_exception, "initial memory out of range");
         if (mt.limits.flags) {
            mt.limits.maximum = parse_varuint32(code);
            EOS_VM_ASSERT(mt.limits.maximum >= mt.limits.initial, wasm_parse_exception, "maximum must be at least minimum");
            EOS_VM_ASSERT(mt.limits.maximum <= 65536u, wasm_parse_exception, "maximum memory out of range");
         }
      }

      void parse_export_entry(wasm_code_ptr& code, export_entry& entry) {
         entry.field_str = parse_utf8_string(code, detail::get_max_symbol_bytes(_options));
         entry.kind  = (external_kind)(*code++);
         entry.index = parse_varuint32(code);
         switch(entry.kind) {
            case external_kind::Function: EOS_VM_ASSERT(entry.index < _mod->get_functions_total(), wasm_parse_exception, "function export out of range"); break;
            case external_kind::Table: EOS_VM_ASSERT(entry.index < _mod->tables.size(), wasm_parse_exception, "table export out of range"); break;
            case external_kind::Memory: EOS_VM_ASSERT(entry.index < _mod->memories.size(), wasm_parse_exception, "memory export out of range"); break;
            case external_kind::Global:
               EOS_VM_ASSERT(entry.index < _mod->globals.size(), wasm_parse_exception, "global export out of range");
               EOS_VM_ASSERT(!detail::get_forbid_export_mutable_globals(_options) || !_mod->globals.at(entry.index).type.mutability,
                             wasm_parse_exception, "cannot export mutable globals");
               break;
            default: EOS_VM_ASSERT(false, wasm_parse_exception, "Unknown export kind"); break;
         }
      }

      void parse_func_type(wasm_code_ptr& code, func_type& ft) {
         ft.form                              = *code++;
         EOS_VM_ASSERT(ft.form == 0x60, wasm_parse_exception, "invalid function type");
         decltype(ft.param_types) param_types(parse_varuint32(code));
         for (size_t i = 0; i < param_types.size(); i++) {
            uint8_t pt        = *code++;
            param_types.at(i) = pt;
            EOS_VM_ASSERT(pt == types::i32 || pt == types::i64 || pt == types::f32 || pt == types::f64 || (pt == types::v128 && detail::get_enable_simd(_options)),
                          wasm_parse_exception, "invalid function param type");
         }
         ft.param_types  = std::move(param_types);
         ft.return_count = *code++;
         EOS_VM_ASSERT(ft.return_count < 2, wasm_parse_exception, "invalid function return count");
         if (ft.return_count > 0) {
            uint8_t rt        = *code++;
            ft.return_type = rt;
            EOS_VM_ASSERT(rt == types::i32 || rt == types::i64 || rt == types::f32 || rt == types::f64 || (rt == types::v128 && detail::get_enable_simd(_options)),
                          wasm_parse_exception, "invalid function return type");
         }
      }

      void normalize_types() {
         type_aliases.resize(_mod->types.size());
         for (uint32_t i = 0; i < _mod->types.size(); ++i) {
            uint32_t j = 0;
            for (; j < i; ++j) {
               if (_mod->types[j] == _mod->types[i]) {
                  break;
               }
            }
            type_aliases[i] = j;
         }

         uint32_t imported_functions_size = _mod->get_imported_functions_size();
         fast_functions.resize(_mod->functions.size() + imported_functions_size);
         for (uint32_t i = 0; i < imported_functions_size; ++i) {
            fast_functions[i] = type_aliases.at(_mod->imports[i].type.func_t);
         }
         for (uint32_t i = 0; i < _mod->functions.size(); ++i) {
            fast_functions[i + imported_functions_size] = type_aliases.at(_mod->functions[i]);
         }
      }

      void parse_elem_segment(wasm_code_ptr& code, elem_segment& es) {
         table_type* tt = nullptr;
         std::uint32_t flags = parse_varuint32(code);
         EOS_VM_ASSERT(es.index == 0 || detail::get_enable_bulk_memory(_options), wasm_parse_exception, "Bulk memory not enabled");
         EOS_VM_ASSERT(es.index <= 7, wasm_parse_exception, "Illegal flags for elem");
         if (flags == 2 || flags == 6) {
            es.index = parse_varuint32(code);
         } else {
            es.index = 0;
         }
         if ((flags & 1) == 0) {
            parse_init_expr(code, es.offset, types::i32);
            es.mode = elem_mode::active;
            EOS_VM_ASSERT(es.index < _mod->tables.size(), wasm_parse_exception, "wrong table in elem");
            tt = &_mod->tables[es.index];
         } else {
            if (flags & 2) {
               es.mode = elem_mode::declarative;
            } else {
               es.mode = elem_mode::passive;
            }
         }
         if (flags == 1 || flags == 2 || flags == 3) {
            auto elemkind = *code++;
            EOS_VM_ASSERT(elemkind == 0x00, wasm_parse_exception, "elemkind must be funcref");
         } else if (flags == 5 || flags == 6 || flags == 7) {
            auto reftype = *code++;
            // externref requires reference types, which we do not support
            EOS_VM_ASSERT(reftype == types::funcref, wasm_parse_exception, "elem type must be funcref");
         }
         EOS_VM_ASSERT(!tt || tt->element_type == types::anyfunc, wasm_parse_exception, "elem type does not match table type");
         uint32_t           size  = parse_varuint32(code);
         EOS_VM_ASSERT(size <= detail::get_max_element_segment_elements(_options), wasm_parse_exception, "elem segment too large");
         decltype(es.elems) elems(size);
         if (flags & 4) {
            for (uint32_t i = 0; i < size; i++) {
               parse_elem_expr(code, elems.at(i));
            }
         } else {
            for (uint32_t i = 0; i < size; i++) {
               uint32_t index    = parse_varuint32(code);
               elems.at(i).type  = fast_functions.at(index);
               elems.at(i).index = index;
               EOS_VM_ASSERT(index < _mod->get_functions_total(), wasm_parse_exception,  "elem for undefined function");
            }
         }
         es.elems = std::move(elems);
      }

      void parse_elem_expr(wasm_code_ptr& code, table_entry& te) {
         auto opcode = *code++;
         switch (opcode) {
            case opcodes::ref_null:
               te.type = te.index = std::numeric_limits<std::uint32_t>::max();
               EOS_VM_ASSERT(*code == types::funcref, wasm_parse_exception, "Unknown type for elem");
               ++code;
               break;
            case opcodes::ref_func:
               te.index = parse_varuint32(code);
               EOS_VM_ASSERT(te.index < _mod->get_functions_total(), wasm_parse_exception, "elem for undefined function");
               te.type = fast_functions.at(te.index);
               break;
            default:
               EOS_VM_ASSERT(false, wasm_parse_exception,
                             "elem expression can only acception ref.null and ref.func");
         }
         EOS_VM_ASSERT((*code++) == opcodes::end, wasm_parse_exception, "no end op found");
      }

      void parse_init_expr(wasm_code_ptr& code, init_expr& ie, uint8_t type) {
         ie.opcode = *code++;
         switch (ie.opcode) {
            case opcodes::i32_const:
               ie.value.i32 = parse_varint32(code);
               EOS_VM_ASSERT(type == types::i32, wasm_parse_exception, "expected i32 initializer");
               break;
            case opcodes::i64_const:
               ie.value.i64 = parse_varint64(code);
               EOS_VM_ASSERT(type == types::i64, wasm_parse_exception, "expected i64 initializer");
               break;
            case opcodes::f32_const:
               ie.value.f32 = parse_raw<uint32_t>(code);
               EOS_VM_ASSERT(type == types::f32, wasm_parse_exception, "expected f32 initializer");
               break;
            case opcodes::f64_const:
               ie.value.f64 = parse_raw<uint64_t>(code);
               EOS_VM_ASSERT(type == types::f64, wasm_parse_exception, "expected f64 initializer");
               break;
            case opcodes::vector_prefix:
               EOS_VM_ASSERT(detail::get_enable_simd(_options), wasm_parse_exception, "SIMD not enabled");
               EOS_VM_ASSERT(parse_varuint32(code) == vec_opcodes::v128_const, wasm_parse_exception, "Expected v128.const");
               ie.value.v128 = parse_v128(code);
               EOS_VM_ASSERT(type == types::v128, wasm_parse_exception, "expected v128 initializer");
               break;
            default:
               EOS_VM_ASSERT(false, wasm_parse_exception,
                             "initializer expression can only acception i32.const, i64.const, f32.const, f64.const and v128.const");
         }
         EOS_VM_ASSERT((*code++) == opcodes::end, wasm_parse_exception, "no end op found");
      }

      void parse_function_body(wasm_code_ptr& code, function_body& fb, std::size_t idx) {
         fb.size   = parse_varuint32(code);
         EOS_VM_ASSERT(fb.size <= detail::get_max_code_bytes(_options), wasm_parse_exception, "Function body too large");
         const auto&         before    = code.offset();
         const auto&         local_cnt = parse_varuint32(code);
         _current_function_index++;
         EOS_VM_ASSERT(local_cnt <= detail::get_max_local_sets(_options), wasm_parse_exception, "Number of local sets exceeds limit");
         decltype(fb.locals) locals(local_cnt);
         func_type& ft = _mod->types.at(_mod->functions.at(idx));
         detail::max_func_local_bytes_checker<Options> local_checker(_options, ft);
         // parse the local entries
         for (size_t i = 0; i < local_cnt; i++) {
            auto count = parse_varuint32(code);
            auto type = *code++;
            if (detail::get_allow_invalid_empty_local_set(_options) && count == 0) type = types::i32;
            EOS_VM_ASSERT(type == types::i32 || type == types::i64 || type == types::f32 || type == types::f64 || (type == types::v128 && detail::get_enable_simd(_options)),
                          wasm_parse_exception, "invalid local type");
            local_checker.on_local(_options, type, count);
            locals.at(i).count = count;
            locals.at(i).type  = type;
         }
         fb.locals = std::move(locals);

         fb.size -= code.offset() - before;
         auto guard = code.scoped_shrink_bounds(fb.size);
         _function_bodies.emplace_back(wasm_code_ptr{code.raw(), fb.size}, local_checker);

         code += fb.size-1;
         EOS_VM_ASSERT(detail::get_allow_code_after_function_end(_options) || *code == 0x0B,
                       wasm_parse_exception, "failed parsing function body, expected 'end'");
         ++code;
      }

      // The control stack holds either address of the target of the
      // label (for backward jumps) or a list of instructions to be
      // updated (for forward jumps).
      //
      // Inside an if: The first element refers to the `if` and should
      // jump to `else`.  The remaining elements should branch to `end`
      using label_t = decltype(std::declval<Writer>().emit_end());
      using branch_t = decltype(std::declval<Writer>().emit_if());
      struct pc_element_t {
         uint32_t operand_depth;
         uint32_t expected_result;
         uint32_t label_result;
         bool is_if;
         std::variant<label_t, std::vector<branch_t>> relocations;
      };

      static constexpr uint8_t any_type = 0x82;
      struct operand_stack_type_tracker {
         explicit operand_stack_type_tracker(const detail::max_func_local_bytes_stack_checker<Options> local_bytes_checker, const Options& options)
          : local_bytes_checker(local_bytes_checker), options(options) {
            stack_usage.update(local_bytes_checker);
         }
         std::vector<uint8_t> state = { scope_tag };
         static constexpr uint8_t unreachable_tag = 0x80;
         static constexpr uint8_t scope_tag = 0x81;
         uint32_t operand_depth = 0;
         uint32_t maximum_operand_depth = 0;
         detail::max_func_local_bytes_stack_checker<Options> local_bytes_checker;
         detail::max_stack_usage_calculator<Options> stack_usage;
         const Options& options;
         void push(uint8_t type) {
            assert(type != unreachable_tag && type != scope_tag);
            assert(type == types::i32 || type == types::i64 || type == types::f32 || type == types::f64 || type == any_type || (type == types::v128 && detail::get_enable_simd(options)));
            EOS_VM_ASSERT(operand_depth < std::numeric_limits<uint32_t>::max(), wasm_parse_exception, "integer overflow in operand depth");
            operand_depth += Writer::get_depth_for_type(type);
            maximum_operand_depth = std::max(operand_depth, maximum_operand_depth);
            state.push_back(type);
            local_bytes_checker.push_stack(options, type);
            stack_usage.update(local_bytes_checker);
         }
         void pop(uint8_t expected) {
            assert(expected != unreachable_tag && expected != scope_tag);
            if(expected == types::pseudo) return;
            EOS_VM_ASSERT(!state.empty(), wasm_parse_exception, "unexpected pop");
            if (state.back() != unreachable_tag) {
               EOS_VM_ASSERT(state.back() == expected || state.back() == any_type, wasm_parse_exception, "wrong type");
               local_bytes_checker.pop_stack(options, expected);
               operand_depth -= Writer::get_depth_for_type(expected);
               state.pop_back();
            }
         }
         uint8_t pop() {
            EOS_VM_ASSERT(!state.empty() && state.back() != scope_tag, wasm_parse_exception, "unexpected pop");
            if (state.back() == unreachable_tag)
               return any_type;
            else {
               uint8_t result = state.back();
               operand_depth -= Writer::get_depth_for_type(result);
               local_bytes_checker.pop_stack(options, result);
               state.pop_back();
               return result;
            }
         }
         void top(uint8_t expected) {
            // Constrain the top of the stack if it was any_type or unreachable_tag.
            pop(expected);
            push(expected);
         }
         void start_unreachable() {
            while(!state.empty() && state.back() != scope_tag) {
               if (state.back() != unreachable_tag)
                  operand_depth -= Writer::get_depth_for_type(state.back());
               state.pop_back();
            }
            local_bytes_checker.push_unreachable();
            state.push_back(unreachable_tag);
         }
         void push_scope() {
            state.push_back(scope_tag);
         }
         void pop_scope(uint8_t expected_result = types::pseudo) {
            pop(expected_result);
            EOS_VM_ASSERT(!state.empty(), wasm_parse_exception, "unexpected end");
            if (state.back() == unreachable_tag) {
               local_bytes_checker.pop_unreachable();
               state.pop_back();
            }
            EOS_VM_ASSERT(state.back() == scope_tag, wasm_parse_exception, "unexpected end");
            state.pop_back();
            if (expected_result != types::pseudo) {
               push(expected_result);
            }
         }
         void finish() {
            if (!state.empty() && state.back() == unreachable_tag) {
               state.pop_back();
            }
            EOS_VM_ASSERT(state.empty(), wasm_parse_exception, "stack not empty at scope end");
         }
         uint32_t depth() const { return operand_depth; }
      };

      struct local_types_t {
         local_types_t(const func_type& ft, const std::vector<local_entry>& locals_arg) :
            _ft(ft), _locals(locals_arg) {
            uint32_t count = ft.param_types.size();
            _boundaries.push_back(count);
            for (uint32_t i = 0; i < locals_arg.size(); ++i) {
               // This test cannot overflow.
               EOS_VM_ASSERT (count <= 0xFFFFFFFFu - locals_arg[i].count, wasm_parse_exception, "too many locals");
               count += locals_arg[i].count;
               _boundaries.push_back(count);
            }
         }
         uint8_t operator[](uint32_t local_idx) const {
            EOS_VM_ASSERT(local_idx < _boundaries.back(), wasm_parse_exception, "undefined local");
            auto pos = std::upper_bound(_boundaries.begin(), _boundaries.end(), local_idx);
            if (pos == _boundaries.begin())
               return _ft.param_types[local_idx];
            else
               return _locals[pos - _boundaries.begin() - 1].type;
         }
         uint64_t locals_count() const {
            uint64_t total = _boundaries.back();
            return total - _ft.param_types.size();
         }
         const func_type& _ft;
         const std::vector<local_entry>& _locals;
         std::vector<uint32_t> _boundaries;
      };

      void parse_function_body_code(wasm_code_ptr& code, size_t bounds, const detail::max_func_local_bytes_stack_checker<Options>& local_bytes_checker,
                                    Writer& code_writer, const func_type& ft, const local_types_t& local_types) {

         // Initialize the control stack with the current function as the sole element
         operand_stack_type_tracker op_stack{local_bytes_checker, _options};
         std::vector<pc_element_t> pc_stack{{
               op_stack.depth(),
               ft.return_count ? ft.return_type : static_cast<uint32_t>(types::pseudo),
               ft.return_count ? ft.return_type : static_cast<uint32_t>(types::pseudo),
               false,
               std::vector<branch_t>{}}};

         // writes the continuation of a label to address.  If the continuation
         // is not yet available, address will be recorded in the relocations
         // list for label.
         auto handle_branch_target = [&](uint32_t label, branch_t address) {
            EOS_VM_ASSERT(label < pc_stack.size(), wasm_parse_exception, "invalid label");
            pc_element_t& branch_target = pc_stack[pc_stack.size() - label - 1];
            std::visit(overloaded{ [&](label_t target) { code_writer.fix_branch(address, target); },
                                   [&](std::vector<branch_t>& relocations) { relocations.push_back(address); } },
               branch_target.relocations);
         };

         // Returns the number of operands that need to be popped when
         // branching to label.  If the label has a return value it will
         // be counted in this, and the high bit will be set to signal
         // its presence.
         auto compute_depth_change = [&](uint32_t label) {
            EOS_VM_ASSERT(label < pc_stack.size(), wasm_parse_exception, "invalid label");
            pc_element_t& branch_target = pc_stack[pc_stack.size() - label - 1];
            auto result = op_stack.depth() - branch_target.operand_depth;
            if(branch_target.label_result != types::pseudo) {
               op_stack.top(branch_target.label_result);
            }
            return std::pair{result, branch_target.label_result};
         };

         // Handles branches to the end of the scope and pops the pc_stack
         auto exit_scope = [&]() {
            // There must be at least one element
            EOS_VM_ASSERT(pc_stack.size(), wasm_parse_exception, "unexpected end instruction");
            // an if with an empty else cannot have a return value
            EOS_VM_ASSERT(!pc_stack.back().is_if || pc_stack.back().expected_result == types::pseudo, wasm_parse_exception, "wrong type");
            auto end_pos = code_writer.emit_end();
            if(auto* relocations = std::get_if<std::vector<branch_t>>(&pc_stack.back().relocations)) {
               for(auto branch_op : *relocations) {
                  code_writer.fix_branch(branch_op, end_pos);
               }
            }
            op_stack.pop_scope(pc_stack.back().expected_result);
            pc_stack.pop_back();
         };

         auto check_in_bounds = [&]{
            EOS_VM_ASSERT(!detail::get_allow_code_after_function_end(_options) || !pc_stack.empty(),
                          wasm_parse_exception, "code after function end");
         };

         while (code.offset() < bounds) {
            EOS_VM_ASSERT(pc_stack.size() <= detail::get_max_nested_structures(_options), wasm_parse_exception,
                          "nested structures validation failure");

            imap.on_instr_start(code_writer.get_addr(), code.raw());

            switch (*code++) {
               case opcodes::unreachable: check_in_bounds(); code_writer.emit_unreachable(); op_stack.start_unreachable(); break;
               case opcodes::nop: code_writer.emit_nop(); break;
               case opcodes::end: {
                  check_in_bounds();
                  exit_scope();
                  EOS_VM_ASSERT(detail::get_allow_code_after_function_end(_options) ||
                                !pc_stack.empty() || code.offset() == bounds, wasm_parse_exception, "function too short");
                  _nested_checker.on_end(_options);
                  break;
               }
               case opcodes::return_: {
                  check_in_bounds();
                  uint32_t label = pc_stack.size() - 1;
                  auto [depth_change,rt] = compute_depth_change(label);
                  auto branch = code_writer.emit_return(depth_change, rt);
                  handle_branch_target(label, branch);
                  op_stack.start_unreachable();
               } break;
               case opcodes::block: {
                  uint32_t expected_result = *code++;
                  if(detail::get_allow_zero_blocktype(_options) && expected_result == 0)
                     expected_result = types::pseudo;
                  EOS_VM_ASSERT(expected_result == types::i32 || expected_result == types::i64 ||
                                expected_result == types::f32 || expected_result == types::f64 ||
                                (expected_result == types::v128 && detail::get_enable_simd(_options)) ||
                                expected_result == types::pseudo, wasm_parse_exception,
                                "Invalid type code in block");
                  pc_stack.push_back({op_stack.depth(), expected_result, expected_result, false, std::vector<branch_t>{}});
                  code_writer.emit_block();
                  op_stack.push_scope();
                  _nested_checker.on_control(_options);
               } break;
               case opcodes::loop: {
                  uint32_t expected_result = *code++;
                  if(detail::get_allow_zero_blocktype(_options) && expected_result == 0)
                     expected_result = types::pseudo;
                  EOS_VM_ASSERT(expected_result == types::i32 || expected_result == types::i64 ||
                                expected_result == types::f32 || expected_result == types::f64 ||
                                (expected_result == types::v128 && detail::get_enable_simd(_options)) ||
                                expected_result == types::pseudo, wasm_parse_exception,
                                "Invalid type code in loop");
                  auto pos = code_writer.emit_loop();
                  pc_stack.push_back({op_stack.depth(), expected_result, types::pseudo, false, pos});
                  op_stack.push_scope();
                  _nested_checker.on_control(_options);
               } break;
               case opcodes::if_: {
                  check_in_bounds();
                  uint32_t expected_result = *code++;
                  if(detail::get_allow_zero_blocktype(_options) && expected_result == 0)
                     expected_result = types::pseudo;
                  EOS_VM_ASSERT(expected_result == types::i32 || expected_result == types::i64 ||
                                expected_result == types::f32 || expected_result == types::f64 ||
                                (expected_result == types::v128 && detail::get_enable_simd(_options)) ||
                                expected_result == types::pseudo, wasm_parse_exception,
                                "Invalid type code in if");
                  auto branch = code_writer.emit_if();
                  op_stack.pop(types::i32);
                  pc_stack.push_back({op_stack.depth(), expected_result, expected_result, true, std::vector{branch}});
                  op_stack.push_scope();
                  _nested_checker.on_control(_options);
               } break;
               case opcodes::else_: {
                  check_in_bounds();
                  auto& old_index = pc_stack.back();
                  EOS_VM_ASSERT(old_index.is_if, wasm_parse_exception, "else outside if");
                  auto& relocations = std::get<std::vector<branch_t>>(old_index.relocations);
                  // reset the operand stack to the same state as the if
                  op_stack.pop(old_index.expected_result);
                  op_stack.pop_scope();
                  op_stack.push_scope();
                  // Overwrite the branch from the `if` with the `else`.
                  // We're left with a normal relocation list where everything
                  // branches to the corresponding `end`
                  relocations[0] = code_writer.emit_else(relocations[0]);
                  old_index.is_if = false;
                  _nested_checker.on_control(_options);
                  break;
               }
               case opcodes::br: {
                  check_in_bounds();
                  uint32_t label = parse_varuint32(code);
                  auto [depth_change,rt] = compute_depth_change(label);
                  auto branch = code_writer.emit_br(depth_change, rt);
                  handle_branch_target(label, branch);
                  op_stack.start_unreachable();
               } break;
               case opcodes::br_if: {
                  check_in_bounds();
                  uint32_t label = parse_varuint32(code);
                  op_stack.pop(types::i32);
                  auto [depth_change,rt] = compute_depth_change(label);
                  auto branch = code_writer.emit_br_if(depth_change, rt);
                  handle_branch_target(label, branch);
               } break;
               case opcodes::br_table: {
                  check_in_bounds();
                  size_t table_size = parse_varuint32(code);
                  EOS_VM_ASSERT(table_size <= detail::get_max_br_table_elements(_options), wasm_parse_exception, "Too many labels in br_table");
                  uint8_t result_type = 0;
                  op_stack.pop(types::i32);
                  auto handler = code_writer.emit_br_table(table_size);
                  for (size_t i = 0; i < table_size; i++) {
                     uint32_t label = parse_varuint32(code);
                     auto [depth_change,rt] = compute_depth_change(label);
                     auto branch = handler.emit_case(depth_change, rt);
                     handle_branch_target(label, branch);
                     uint8_t one_result = pc_stack[pc_stack.size() - label - 1].label_result;
                     if(i == 0) {
                        result_type = one_result;
                     } else {
                        EOS_VM_ASSERT(result_type == one_result, wasm_parse_exception, "br_table labels must have the same type");
                     }
                  }
                  uint32_t label = parse_varuint32(code);
                  auto [depth_change,rt] = compute_depth_change(label);
                  auto branch = handler.emit_default(depth_change, rt);
                  handle_branch_target(label, branch);
                  EOS_VM_ASSERT(table_size == 0 || result_type == pc_stack[pc_stack.size() - label - 1].label_result,
                                wasm_parse_exception, "br_table labels must have the same type");
                  op_stack.start_unreachable();
               } break;
               case opcodes::call: {
                  check_in_bounds();
                  uint32_t funcnum = parse_varuint32(code);
                  const func_type& ft = _mod->get_function_type(funcnum);
                  for(uint32_t i = 0; i < ft.param_types.size(); ++i)
                     op_stack.pop(ft.param_types[ft.param_types.size() - i - 1]);
                  EOS_VM_ASSERT(ft.return_count <= 1, wasm_parse_exception, "unsupported");
                  if(ft.return_count)
                     op_stack.push(ft.return_type);
                  code_writer.emit_call(ft, funcnum);
               } break;
               case opcodes::call_indirect: {
                  check_in_bounds();
                  uint32_t functypeidx = parse_varuint32(code);
                  const func_type& ft = _mod->types.at(functypeidx);
                  EOS_VM_ASSERT(_mod->tables.size() > 0, wasm_parse_exception, "call_indirect requires a table");
                  op_stack.pop(types::i32);
                  for(uint32_t i = 0; i < ft.param_types.size(); ++i)
                     op_stack.pop(ft.param_types[ft.param_types.size() - i - 1]);
                  EOS_VM_ASSERT(ft.return_count <= 1, wasm_parse_exception, "unsupported");
                  if(ft.return_count)
                     op_stack.push(ft.return_type);
                  code_writer.emit_call_indirect(ft, type_aliases[functypeidx]);
                  EOS_VM_ASSERT(*code == 0, wasm_parse_exception, "call_indirect must end with 0x00.");
                  code++; // 0x00
                  break;
               }
               case opcodes::drop: check_in_bounds(); code_writer.emit_drop(op_stack.pop()); break;
               case opcodes::select: {
                  check_in_bounds();
                  op_stack.pop(types::i32);
                  uint8_t t0 = op_stack.pop();
                  uint8_t t1 = op_stack.pop();
                  EOS_VM_ASSERT(t0 == t1 || t0 == any_type || t1 == any_type, wasm_parse_exception, "incorrect types for select");
                  op_stack.push(t0 != any_type? t0 : t1);
                  code_writer.emit_select(t0);
               } break;
               case opcodes::get_local: {
                  uint32_t local_idx = parse_varuint32(code);
                  op_stack.push(local_types[local_idx]);
                  code_writer.emit_get_local(local_idx, local_types[local_idx]);
               } break;
               case opcodes::set_local: {
                  check_in_bounds();
                  uint32_t local_idx = parse_varuint32(code);
                  op_stack.pop(local_types[local_idx]);
                  code_writer.emit_set_local(local_idx, local_types[local_idx]);
               } break;
               case opcodes::tee_local: {
                  check_in_bounds();
                  uint32_t local_idx = parse_varuint32(code);
                  op_stack.top(local_types[local_idx]);
                  code_writer.emit_tee_local(local_idx, local_types[local_idx]);
               } break;
               case opcodes::get_global: {
                  uint32_t global_idx = parse_varuint32(code);
                  op_stack.push(_mod->globals.at(global_idx).type.content_type);
                  code_writer.emit_get_global(global_idx);
               } break;
               case opcodes::set_global: {
                  check_in_bounds();
                  uint32_t global_idx = parse_varuint32(code);
                  EOS_VM_ASSERT(_mod->globals.at(global_idx).type.mutability, wasm_parse_exception, "cannot set const global");
                  op_stack.pop(_mod->globals.at(global_idx).type.content_type);
                  code_writer.emit_set_global(global_idx);
               } break;
#define LOAD_OP(op_name, max_align, type)                            \
               case opcodes::op_name: {                              \
                  check_in_bounds();                                 \
                  EOS_VM_ASSERT(_mod->memories.size() > 0, wasm_parse_exception, "load requires memory"); \
                  uint32_t alignment = parse_varuint32(code);        \
                  uint32_t offset = parse_varuint32(code);           \
                  EOS_VM_ASSERT(alignment <= uint32_t(max_align), wasm_parse_exception, "alignment cannot be greater than size."); \
                  EOS_VM_ASSERT(offset <= detail::get_max_memory_offset(_options), wasm_parse_exception, "load offset too large."); \
                  op_stack.pop(types::i32);                          \
                  op_stack.push(types::type);                        \
                  code_writer.emit_ ## op_name( alignment, offset ); \
               } break;

               LOAD_OP(i32_load, 2, i32)
               LOAD_OP(i64_load, 3, i64)
               LOAD_OP(f32_load, 2, f32)
               LOAD_OP(f64_load, 3, f64)
               LOAD_OP(i32_load8_s, 0, i32)
               LOAD_OP(i32_load16_s, 1, i32)
               LOAD_OP(i32_load8_u, 0, i32)
               LOAD_OP(i32_load16_u, 1, i32)
               LOAD_OP(i64_load8_s, 0, i64)
               LOAD_OP(i64_load16_s, 1, i64)
               LOAD_OP(i64_load32_s, 2, i64)
               LOAD_OP(i64_load8_u, 0, i64)
               LOAD_OP(i64_load16_u, 1, i64)
               LOAD_OP(i64_load32_u, 2, i64)

#define STORE_OP(op_name, max_align, type)                           \
               case opcodes::op_name: {                              \
                  check_in_bounds();                                 \
                  EOS_VM_ASSERT(_mod->memories.size() > 0, wasm_parse_exception, "store requires memory"); \
                  uint32_t alignment = parse_varuint32(code);        \
                  uint32_t offset = parse_varuint32(code);           \
                  EOS_VM_ASSERT(alignment <= uint32_t(max_align), wasm_parse_exception, "alignment cannot be greater than size."); \
                  EOS_VM_ASSERT(offset <= detail::get_max_memory_offset(_options), wasm_parse_exception, "store offset too large."); \
                  op_stack.pop(types::type);                         \
                  op_stack.pop(types::i32);                          \
                  code_writer.emit_ ## op_name( alignment, offset ); \
               } break;

               STORE_OP(i32_store, 2, i32)
               STORE_OP(i64_store, 3, i64)
               STORE_OP(f32_store, 2, f32)
               STORE_OP(f64_store, 3, f64)
               STORE_OP(i32_store8, 0, i32)
               STORE_OP(i32_store16, 1, i32)
               STORE_OP(i64_store8, 0, i64)
               STORE_OP(i64_store16, 1, i64)
               STORE_OP(i64_store32, 2, i64)

               case opcodes::current_memory:
                  EOS_VM_ASSERT(_mod->memories.size() != 0, wasm_parse_exception, "memory.size requires memory");
                  op_stack.push(types::i32);
                  EOS_VM_ASSERT(*code == 0, wasm_parse_exception, "memory.size must end with 0x00");
                  code++;
                  code_writer.emit_current_memory();
                  break;
               case opcodes::grow_memory:
                  check_in_bounds();
                  EOS_VM_ASSERT(_mod->memories.size() != 0, wasm_parse_exception, "memory.grow requires memory");
                  op_stack.pop(types::i32);
                  op_stack.push(types::i32);
                  EOS_VM_ASSERT(*code == 0, wasm_parse_exception, "memory.grow must end with 0x00");
                  code++;
                  code_writer.emit_grow_memory();
                  break;
               case opcodes::i32_const: code_writer.emit_i32_const( parse_varint32(code) ); op_stack.push(types::i32); break;
               case opcodes::i64_const: code_writer.emit_i64_const( parse_varint64(code) ); op_stack.push(types::i64); break;
               case opcodes::f32_const: {
                  code_writer.emit_f32_const( parse_raw<float>(code) );
                  op_stack.push(types::f32);
               } break;
               case opcodes::f64_const: {
                  code_writer.emit_f64_const( parse_raw<double>(code) );
                  op_stack.push(types::f64);
               } break;

#define UNOP(opname) \
               case opcodes::opname: check_in_bounds(); code_writer.emit_ ## opname(); op_stack.pop(types::A); op_stack.push(types::R); break;
#define BINOP(opname) \
               case opcodes::opname: check_in_bounds(); code_writer.emit_ ## opname(); op_stack.pop(types::A); op_stack.pop(types::A); op_stack.push(types::R); break;
#define CASTOP(dst, opname, src)                                         \
               case opcodes::dst ## _ ## opname ## _ ## src: check_in_bounds(); code_writer.emit_ ## dst ## _ ## opname ## _ ## src(); op_stack.pop(types::src); op_stack.push(types::dst); break;

#define R i32
#define A i32
               UNOP(i32_eqz)
               BINOP(i32_eq)
               BINOP(i32_ne)
               BINOP(i32_lt_s)
               BINOP(i32_lt_u)
               BINOP(i32_gt_s)
               BINOP(i32_gt_u)
               BINOP(i32_le_s)
               BINOP(i32_le_u)
               BINOP(i32_ge_s)
               BINOP(i32_ge_u)
#undef A
#define A i64
               UNOP(i64_eqz)
               BINOP(i64_eq)
               BINOP(i64_ne)
               BINOP(i64_lt_s)
               BINOP(i64_lt_u)
               BINOP(i64_gt_s)
               BINOP(i64_gt_u)
               BINOP(i64_le_s)
               BINOP(i64_le_u)
               BINOP(i64_ge_s)
               BINOP(i64_ge_u)
#undef A
#define A f32
               BINOP(f32_eq)
               BINOP(f32_ne)
               BINOP(f32_lt)
               BINOP(f32_gt)
               BINOP(f32_le)
               BINOP(f32_ge)
#undef A
#define A f64
               BINOP(f64_eq)
               BINOP(f64_ne)
               BINOP(f64_lt)
               BINOP(f64_gt)
               BINOP(f64_le)
               BINOP(f64_ge)
#undef A
#undef R
#define R A
#define A i32
               UNOP(i32_clz)
               UNOP(i32_ctz)
               UNOP(i32_popcnt)
               BINOP(i32_add)
               BINOP(i32_sub)
               BINOP(i32_mul)
               BINOP(i32_div_s)
               BINOP(i32_div_u)
               BINOP(i32_rem_s)
               BINOP(i32_rem_u)
               BINOP(i32_and)
               BINOP(i32_or)
               BINOP(i32_xor)
               BINOP(i32_shl)
               BINOP(i32_shr_s)
               BINOP(i32_shr_u)
               BINOP(i32_rotl)
               BINOP(i32_rotr)
#undef A
#define A i64
               UNOP(i64_clz)
               UNOP(i64_ctz)
               UNOP(i64_popcnt)
               BINOP(i64_add)
               BINOP(i64_sub)
               BINOP(i64_mul)
               BINOP(i64_div_s)
               BINOP(i64_div_u)
               BINOP(i64_rem_s)
               BINOP(i64_rem_u)
               BINOP(i64_and)
               BINOP(i64_or)
               BINOP(i64_xor)
               BINOP(i64_shl)
               BINOP(i64_shr_s)
               BINOP(i64_shr_u)
               BINOP(i64_rotl)
               BINOP(i64_rotr)
#undef A
#define A f32
               UNOP(f32_abs)
               UNOP(f32_neg)
               UNOP(f32_ceil)
               UNOP(f32_floor)
               UNOP(f32_trunc)
               UNOP(f32_nearest)
               UNOP(f32_sqrt)
               BINOP(f32_add)
               BINOP(f32_sub)
               BINOP(f32_mul)
               BINOP(f32_div)
               BINOP(f32_min)
               BINOP(f32_max)
               BINOP(f32_copysign)
#undef A
#define A f64
               UNOP(f64_abs)
               UNOP(f64_neg)
               UNOP(f64_ceil)
               UNOP(f64_floor)
               UNOP(f64_trunc)
               UNOP(f64_nearest)
               UNOP(f64_sqrt)
               BINOP(f64_add)
               BINOP(f64_sub)
               BINOP(f64_mul)
               BINOP(f64_div)
               BINOP(f64_min)
               BINOP(f64_max)
               BINOP(f64_copysign)
#undef A
#undef R

               CASTOP(i32, wrap, i64)
               CASTOP(i32, trunc_s, f32)
               CASTOP(i32, trunc_u, f32)
               CASTOP(i32, trunc_s, f64)
               CASTOP(i32, trunc_u, f64)
               CASTOP(i64, extend_s, i32)
               CASTOP(i64, extend_u, i32)
               CASTOP(i64, trunc_s, f32)
               CASTOP(i64, trunc_u, f32)
               CASTOP(i64, trunc_s, f64)
               CASTOP(i64, trunc_u, f64)
               CASTOP(f32, convert_s, i32)
               CASTOP(f32, convert_u, i32)
               CASTOP(f32, convert_s, i64)
               CASTOP(f32, convert_u, i64)
               CASTOP(f32, demote, f64)
               CASTOP(f64, convert_s, i32)
               CASTOP(f64, convert_u, i32)
               CASTOP(f64, convert_s, i64)
               CASTOP(f64, convert_u, i64)
               CASTOP(f64, promote, f32)
               CASTOP(i32, reinterpret, f32)
               CASTOP(i64, reinterpret, f64)
               CASTOP(f32, reinterpret, i32)
               CASTOP(f64, reinterpret, i64)

#undef CASTOP
#undef UNOP
#undef BINOP
                   
#define EXTENDOP(dst, opname)                                           \
               case opcodes::dst ## _ ## opname:                        \
                  check_in_bounds();                                    \
                  EOS_VM_ASSERT(detail::get_enable_sign_ext(_options), wasm_parse_exception, "Sign-extension operators not enabled"); \
                  code_writer.emit_ ## dst ## _ ## opname();            \
                  op_stack.pop(types::dst);                             \
                  op_stack.push(types::dst);                            \
                  break;

               EXTENDOP(i32, extend8_s)
               EXTENDOP(i32, extend16_s)
               EXTENDOP(i64, extend8_s)
               EXTENDOP(i64, extend16_s)
               EXTENDOP(i64, extend32_s)

               case opcodes::vector_prefix: {
                  EOS_VM_ASSERT(detail::get_enable_simd(_options), wasm_parse_exception, "SIMD not enabled");
                  switch(parse_varuint32(code))
                  {
#define opcodes vec_opcodes
                     LOAD_OP(v128_load, 4, v128)
                     LOAD_OP(v128_load8x8_s, 3, v128)
                     LOAD_OP(v128_load8x8_u, 3, v128)
                     LOAD_OP(v128_load16x4_s, 3, v128)
                     LOAD_OP(v128_load16x4_u, 3, v128)
                     LOAD_OP(v128_load32x2_s, 3, v128)
                     LOAD_OP(v128_load32x2_u, 3, v128)
                     LOAD_OP(v128_load8_splat, 0, v128)
                     LOAD_OP(v128_load16_splat, 1, v128)
                     LOAD_OP(v128_load32_splat, 2, v128)
                     LOAD_OP(v128_load64_splat, 3, v128)
                     LOAD_OP(v128_load32_zero, 2, v128)
                     LOAD_OP(v128_load64_zero, 3, v128)
                     STORE_OP(v128_store, 4, v128)
#undef opcodes

#undef LOAD_OP
#undef STORE_OP

#define LOADLANE_OP(op_name, max_align, type)                           \
                     case vec_opcodes::op_name: {                       \
                        check_in_bounds();                              \
                        EOS_VM_ASSERT(_mod->memories.size() > 0, wasm_parse_exception, "load requires memory"); \
                        uint32_t alignment = parse_varuint32(code);     \
                        uint32_t offset = parse_varuint32(code);        \
                        uint8_t lane = *code++;                         \
                        EOS_VM_ASSERT(alignment <= uint32_t(max_align), wasm_parse_exception, "alignment cannot be greater than size."); \
                        EOS_VM_ASSERT(offset <= detail::get_max_memory_offset(_options), wasm_parse_exception, "load offset too large."); \
                        EOS_VM_ASSERT(lane < (1 << (4-max_align)), wasm_parse_exception, "laneidx out of bounds"); \
                        op_stack.pop(types::type);                      \
                        op_stack.pop(types::i32);                       \
                        op_stack.push(types::type);                     \
                        code_writer.emit_ ## op_name( alignment, offset, lane ); \
                     } break;

                     LOADLANE_OP(v128_load8_lane, 0, v128)
                     LOADLANE_OP(v128_load16_lane, 1, v128)
                     LOADLANE_OP(v128_load32_lane, 2, v128)
                     LOADLANE_OP(v128_load64_lane, 3, v128)

#undef LOADLANE_OP

#define STORELANE_OP(op_name, max_align, type)                          \
                     case vec_opcodes::op_name: {                       \
                        check_in_bounds();                              \
                        EOS_VM_ASSERT(_mod->memories.size() > 0, wasm_parse_exception, "load requires memory"); \
                        uint32_t alignment = parse_varuint32(code);     \
                        uint32_t offset = parse_varuint32(code);        \
                        uint8_t lane = *code++;                         \
                        EOS_VM_ASSERT(alignment <= uint32_t(max_align), wasm_parse_exception, "alignment cannot be greater than size."); \
                        EOS_VM_ASSERT(offset <= detail::get_max_memory_offset(_options), wasm_parse_exception, "load offset too large."); \
                        EOS_VM_ASSERT(lane < (1 << (4-max_align)), wasm_parse_exception, "laneidx out of bounds"); \
                        op_stack.pop(types::type);                      \
                        op_stack.pop(types::i32);                       \
                        code_writer.emit_ ## op_name( alignment, offset, lane ); \
                     } break;

                     STORELANE_OP(v128_store8_lane, 0, v128)
                     STORELANE_OP(v128_store16_lane, 1, v128)
                     STORELANE_OP(v128_store32_lane, 2, v128)
                     STORELANE_OP(v128_store64_lane, 3, v128)

#undef STORELANE_OP

                     case vec_opcodes::v128_const: {
                        check_in_bounds();
                        code_writer.emit_v128_const(parse_v128(code));
                        op_stack.push(types::v128);
                        break;
                     }

                     case vec_opcodes::i8x16_shuffle: {
                        check_in_bounds();
                        uint8_t* lanes = code.raw();
                        code += 16;
                        for(int i = 0; i < 16; ++i)
                        {
                           EOS_VM_ASSERT(lanes[i] < 32, wasm_parse_exception, "shuffle laneidx must be less than 32");
                        }
                        code_writer.emit_i8x16_shuffle(lanes);
                        op_stack.pop(types::v128);
                        op_stack.pop(types::v128);
                        op_stack.push(types::v128);
                     } break;

#define EXTRACT_LANE_OP(opcode, type, N)                                \
                     case vec_opcodes::opcode: {                        \
                        check_in_bounds();                              \
                        uint8_t laneidx = *code++;                      \
                        EOS_VM_ASSERT(laneidx < N, wasm_parse_exception, "laneidx must be smaller than dim(shape)"); \
                        op_stack.pop(types::v128);                      \
                        op_stack.push(types::type);                     \
                        code_writer.emit_ ## opcode(laneidx);           \
                     } break;

#define REPLACE_LANE_OP(opcode, type, N)                                \
                     case vec_opcodes::opcode: {                        \
                        check_in_bounds();                              \
                        uint8_t laneidx = *code++;                      \
                        EOS_VM_ASSERT(laneidx < N, wasm_parse_exception, "laneidx must be smaller than dim(shape)"); \
                        op_stack.pop(types::type);                      \
                        op_stack.pop(types::v128);                      \
                        op_stack.push(types::v128);                     \
                        code_writer.emit_ ## opcode(laneidx);           \
                     } break;

                     EXTRACT_LANE_OP(i8x16_extract_lane_s, i32, 16)
                     EXTRACT_LANE_OP(i8x16_extract_lane_u, i32, 16)
                     REPLACE_LANE_OP(i8x16_replace_lane, i32, 16)
                     EXTRACT_LANE_OP(i16x8_extract_lane_s, i32, 8)
                     EXTRACT_LANE_OP(i16x8_extract_lane_u, i32, 8)
                     REPLACE_LANE_OP(i16x8_replace_lane, i32, 8)
                     EXTRACT_LANE_OP(i32x4_extract_lane, i32, 4)
                     REPLACE_LANE_OP(i32x4_replace_lane, i32, 4)
                     EXTRACT_LANE_OP(i64x2_extract_lane, i64, 2)
                     REPLACE_LANE_OP(i64x2_replace_lane, i64, 2)
                     EXTRACT_LANE_OP(f32x4_extract_lane, f32, 4)
                     REPLACE_LANE_OP(f32x4_replace_lane, f32, 4)
                     EXTRACT_LANE_OP(f64x2_extract_lane, f64, 2)
                     REPLACE_LANE_OP(f64x2_replace_lane, f64, 2)

#undef EXTRACT_LANE_OP

#define INPUTS_0()
#define INPUTS_1(t0) op_stack.pop(types::t0);
#define INPUTS_2(t0, t1) op_stack.pop(types::t1); INPUTS_1(t0);
#define INPUTS_3(t0, t1, t2) op_stack.pop(types::t2); INPUTS_2(t1, t0);

#define OUTPUTS_0()
#define OUTPUTS_1(t0) op_stack.push(types::t0);

#define CAT_I(x, y) x ## y
#define CAT(x, y) CAT_I(x, y)
#define VA_SZ_EMPTY() ~,~,~,~
#define VA_SZ_II(a0, a1, a2, a3, n, ...) n
#define VA_SZ_I(...) VA_SZ_II(__VA_ARGS__, 0, 3, 2, 1, 0, ~)
#define VA_SZ(...) VA_SZ_I( VA_SZ_EMPTY __VA_ARGS__ () )

#define NUMERIC_OP(opcode, inputs, outputs)                             \
                     case vec_opcodes::opcode: {                        \
                        check_in_bounds();                              \
                        CAT(INPUTS_, VA_SZ inputs) inputs               \
                        CAT(OUTPUTS_, VA_SZ outputs) outputs            \
                        code_writer.emit_ ## opcode();                  \
                     } break;

                     NUMERIC_OP(i8x16_swizzle, (v128, v128), (v128))
                     NUMERIC_OP(i8x16_splat, (i32), (v128))
                     NUMERIC_OP(i16x8_splat, (i32), (v128))
                     NUMERIC_OP(i32x4_splat, (i32), (v128))
                     NUMERIC_OP(i64x2_splat, (i64), (v128))
                     NUMERIC_OP(f32x4_splat, (f32), (v128))
                     NUMERIC_OP(f64x2_splat, (f64), (v128))

                     NUMERIC_OP(i8x16_eq, (v128, v128), (v128))
                     NUMERIC_OP(i8x16_ne, (v128, v128), (v128))
                     NUMERIC_OP(i8x16_lt_s, (v128, v128), (v128))
                     NUMERIC_OP(i8x16_lt_u, (v128, v128), (v128))
                     NUMERIC_OP(i8x16_gt_s, (v128, v128), (v128))
                     NUMERIC_OP(i8x16_gt_u, (v128, v128), (v128))
                     NUMERIC_OP(i8x16_le_s, (v128, v128), (v128))
                     NUMERIC_OP(i8x16_le_u, (v128, v128), (v128))
                     NUMERIC_OP(i8x16_ge_s, (v128, v128), (v128))
                     NUMERIC_OP(i8x16_ge_u, (v128, v128), (v128))

                     NUMERIC_OP(i16x8_eq, (v128, v128), (v128))
                     NUMERIC_OP(i16x8_ne, (v128, v128), (v128))
                     NUMERIC_OP(i16x8_lt_s, (v128, v128), (v128))
                     NUMERIC_OP(i16x8_lt_u, (v128, v128), (v128))
                     NUMERIC_OP(i16x8_gt_s, (v128, v128), (v128))
                     NUMERIC_OP(i16x8_gt_u, (v128, v128), (v128))
                     NUMERIC_OP(i16x8_le_s, (v128, v128), (v128))
                     NUMERIC_OP(i16x8_le_u, (v128, v128), (v128))
                     NUMERIC_OP(i16x8_ge_s, (v128, v128), (v128))
                     NUMERIC_OP(i16x8_ge_u, (v128, v128), (v128))

                     NUMERIC_OP(i32x4_eq, (v128, v128), (v128))
                     NUMERIC_OP(i32x4_ne, (v128, v128), (v128))
                     NUMERIC_OP(i32x4_lt_s, (v128, v128), (v128))
                     NUMERIC_OP(i32x4_lt_u, (v128, v128), (v128))
                     NUMERIC_OP(i32x4_gt_s, (v128, v128), (v128))
                     NUMERIC_OP(i32x4_gt_u, (v128, v128), (v128))
                     NUMERIC_OP(i32x4_le_s, (v128, v128), (v128))
                     NUMERIC_OP(i32x4_le_u, (v128, v128), (v128))
                     NUMERIC_OP(i32x4_ge_s, (v128, v128), (v128))
                     NUMERIC_OP(i32x4_ge_u, (v128, v128), (v128))

                     NUMERIC_OP(i64x2_eq, (v128, v128), (v128))
                     NUMERIC_OP(i64x2_ne, (v128, v128), (v128))
                     NUMERIC_OP(i64x2_lt_s, (v128, v128), (v128))
                     NUMERIC_OP(i64x2_gt_s, (v128, v128), (v128))
                     NUMERIC_OP(i64x2_le_s, (v128, v128), (v128))
                     NUMERIC_OP(i64x2_ge_s, (v128, v128), (v128))

                     NUMERIC_OP(f32x4_eq, (v128, v128), (v128))
                     NUMERIC_OP(f32x4_ne, (v128, v128), (v128))
                     NUMERIC_OP(f32x4_lt, (v128, v128), (v128))
                     NUMERIC_OP(f32x4_gt, (v128, v128), (v128))
                     NUMERIC_OP(f32x4_le, (v128, v128), (v128))
                     NUMERIC_OP(f32x4_ge, (v128, v128), (v128))

                     NUMERIC_OP(f64x2_eq, (v128, v128), (v128))
                     NUMERIC_OP(f64x2_ne, (v128, v128), (v128))
                     NUMERIC_OP(f64x2_lt, (v128, v128), (v128))
                     NUMERIC_OP(f64x2_gt, (v128, v128), (v128))
                     NUMERIC_OP(f64x2_le, (v128, v128), (v128))
                     NUMERIC_OP(f64x2_ge, (v128, v128), (v128))

                     NUMERIC_OP(v128_not, (v128), (v128));
                     NUMERIC_OP(v128_and, (v128, v128), (v128));
                     NUMERIC_OP(v128_andnot, (v128, v128), (v128));
                     NUMERIC_OP(v128_or, (v128, v128), (v128));
                     NUMERIC_OP(v128_xor, (v128, v128), (v128));
                     NUMERIC_OP(v128_bitselect, (v128, v128, v128), (v128));
                     NUMERIC_OP(v128_any_true, (v128), (i32));

                     NUMERIC_OP(i8x16_abs, (v128), (v128));
                     NUMERIC_OP(i8x16_neg, (v128), (v128));
                     NUMERIC_OP(i8x16_popcnt, (v128), (v128));
                     NUMERIC_OP(i8x16_all_true, (v128), (i32));
                     NUMERIC_OP(i8x16_bitmask, (v128), (i32));
                     NUMERIC_OP(i8x16_narrow_i16x8_s, (v128, v128), (v128));
                     NUMERIC_OP(i8x16_narrow_i16x8_u, (v128, v128), (v128));
                     NUMERIC_OP(i8x16_shl, (v128, i32), (v128));
                     NUMERIC_OP(i8x16_shr_s, (v128, i32), (v128));
                     NUMERIC_OP(i8x16_shr_u, (v128, i32), (v128));
                     NUMERIC_OP(i8x16_add, (v128, v128), (v128));
                     NUMERIC_OP(i8x16_add_sat_s, (v128, v128), (v128));
                     NUMERIC_OP(i8x16_add_sat_u, (v128, v128), (v128));
                     NUMERIC_OP(i8x16_sub, (v128, v128), (v128));
                     NUMERIC_OP(i8x16_sub_sat_s, (v128, v128), (v128));
                     NUMERIC_OP(i8x16_sub_sat_u, (v128, v128), (v128));
                     NUMERIC_OP(i8x16_min_s, (v128, v128), (v128));
                     NUMERIC_OP(i8x16_min_u, (v128, v128), (v128));
                     NUMERIC_OP(i8x16_max_s, (v128, v128), (v128));
                     NUMERIC_OP(i8x16_max_u, (v128, v128), (v128));
                     NUMERIC_OP(i8x16_avgr_u, (v128, v128), (v128));

                     NUMERIC_OP(i16x8_extadd_pairwise_i8x16_s, (v128), (v128));
                     NUMERIC_OP(i16x8_extadd_pairwise_i8x16_u, (v128), (v128));
                     NUMERIC_OP(i16x8_abs, (v128), (v128));
                     NUMERIC_OP(i16x8_neg, (v128), (v128));
                     NUMERIC_OP(i16x8_q15mulr_sat_s, (v128, v128), (v128));
                     NUMERIC_OP(i16x8_all_true, (v128), (i32));
                     NUMERIC_OP(i16x8_bitmask, (v128), (i32));
                     NUMERIC_OP(i16x8_narrow_i32x4_s, (v128, v128), (v128));
                     NUMERIC_OP(i16x8_narrow_i32x4_u, (v128, v128), (v128));
                     NUMERIC_OP(i16x8_extend_low_i8x16_s, (v128), (v128));
                     NUMERIC_OP(i16x8_extend_high_i8x16_s, (v128), (v128));
                     NUMERIC_OP(i16x8_extend_low_i8x16_u, (v128), (v128));
                     NUMERIC_OP(i16x8_extend_high_i8x16_u, (v128), (v128));
                     NUMERIC_OP(i16x8_shl, (v128, i32), (v128));
                     NUMERIC_OP(i16x8_shr_s, (v128, i32), (v128));
                     NUMERIC_OP(i16x8_shr_u, (v128, i32), (v128));
                     NUMERIC_OP(i16x8_add, (v128, v128), (v128));
                     NUMERIC_OP(i16x8_add_sat_s, (v128, v128), (v128));
                     NUMERIC_OP(i16x8_add_sat_u, (v128, v128), (v128));
                     NUMERIC_OP(i16x8_sub, (v128, v128), (v128));
                     NUMERIC_OP(i16x8_sub_sat_s, (v128, v128), (v128));
                     NUMERIC_OP(i16x8_sub_sat_u, (v128, v128), (v128));
                     NUMERIC_OP(i16x8_mul, (v128, v128), (v128));
                     NUMERIC_OP(i16x8_min_s, (v128, v128), (v128));
                     NUMERIC_OP(i16x8_min_u, (v128, v128), (v128));
                     NUMERIC_OP(i16x8_max_s, (v128, v128), (v128));
                     NUMERIC_OP(i16x8_max_u, (v128, v128), (v128));
                     NUMERIC_OP(i16x8_avgr_u, (v128, v128), (v128));
                     NUMERIC_OP(i16x8_extmul_low_i8x16_s, (v128, v128), (v128));
                     NUMERIC_OP(i16x8_extmul_high_i8x16_s, (v128, v128), (v128));
                     NUMERIC_OP(i16x8_extmul_low_i8x16_u, (v128, v128), (v128));
                     NUMERIC_OP(i16x8_extmul_high_i8x16_u, (v128, v128), (v128));

                     NUMERIC_OP(i32x4_extadd_pairwise_i16x8_s, (v128), (v128));
                     NUMERIC_OP(i32x4_extadd_pairwise_i16x8_u, (v128), (v128));
                     NUMERIC_OP(i32x4_abs, (v128), (v128));
                     NUMERIC_OP(i32x4_neg, (v128), (v128));
                     NUMERIC_OP(i32x4_all_true, (v128), (i32));
                     NUMERIC_OP(i32x4_bitmask, (v128), (i32));
                     NUMERIC_OP(i32x4_extend_low_i16x8_s, (v128), (v128));
                     NUMERIC_OP(i32x4_extend_high_i16x8_s, (v128), (v128));
                     NUMERIC_OP(i32x4_extend_low_i16x8_u, (v128), (v128));
                     NUMERIC_OP(i32x4_extend_high_i16x8_u, (v128), (v128));
                     NUMERIC_OP(i32x4_shl, (v128, i32), (v128));
                     NUMERIC_OP(i32x4_shr_s, (v128, i32), (v128));
                     NUMERIC_OP(i32x4_shr_u, (v128, i32), (v128));
                     NUMERIC_OP(i32x4_add, (v128, v128), (v128));
                     NUMERIC_OP(i32x4_sub, (v128, v128), (v128));
                     NUMERIC_OP(i32x4_mul, (v128, v128), (v128));
                     NUMERIC_OP(i32x4_min_s, (v128, v128), (v128));
                     NUMERIC_OP(i32x4_min_u, (v128, v128), (v128));
                     NUMERIC_OP(i32x4_max_s, (v128, v128), (v128));
                     NUMERIC_OP(i32x4_max_u, (v128, v128), (v128));
                     NUMERIC_OP(i32x4_dot_i16x8_s, (v128, v128), (v128));
                     NUMERIC_OP(i32x4_extmul_low_i16x8_s, (v128, v128), (v128));
                     NUMERIC_OP(i32x4_extmul_high_i16x8_s, (v128, v128), (v128));
                     NUMERIC_OP(i32x4_extmul_low_i16x8_u, (v128, v128), (v128));
                     NUMERIC_OP(i32x4_extmul_high_i16x8_u, (v128, v128), (v128));

                     NUMERIC_OP(i64x2_abs, (v128), (v128));
                     NUMERIC_OP(i64x2_neg, (v128), (v128));
                     NUMERIC_OP(i64x2_all_true, (v128), (i32));
                     NUMERIC_OP(i64x2_bitmask, (v128), (i32));
                     NUMERIC_OP(i64x2_extend_low_i32x4_s, (v128), (v128));
                     NUMERIC_OP(i64x2_extend_high_i32x4_s, (v128), (v128));
                     NUMERIC_OP(i64x2_extend_low_i32x4_u, (v128), (v128));
                     NUMERIC_OP(i64x2_extend_high_i32x4_u, (v128), (v128));
                     NUMERIC_OP(i64x2_shl, (v128, i32), (v128));
                     NUMERIC_OP(i64x2_shr_s, (v128, i32), (v128));
                     NUMERIC_OP(i64x2_shr_u, (v128, i32), (v128));
                     NUMERIC_OP(i64x2_add, (v128, v128), (v128));
                     NUMERIC_OP(i64x2_sub, (v128, v128), (v128));
                     NUMERIC_OP(i64x2_mul, (v128, v128), (v128));
                     NUMERIC_OP(i64x2_extmul_low_i32x4_s, (v128, v128), (v128));
                     NUMERIC_OP(i64x2_extmul_high_i32x4_s, (v128, v128), (v128));
                     NUMERIC_OP(i64x2_extmul_low_i32x4_u, (v128, v128), (v128));
                     NUMERIC_OP(i64x2_extmul_high_i32x4_u, (v128, v128), (v128));

                     NUMERIC_OP(f32x4_ceil, (v128), (v128));
                     NUMERIC_OP(f32x4_floor, (v128), (v128));
                     NUMERIC_OP(f32x4_trunc, (v128), (v128));
                     NUMERIC_OP(f32x4_nearest, (v128), (v128));
                     NUMERIC_OP(f32x4_abs, (v128), (v128));
                     NUMERIC_OP(f32x4_neg, (v128), (v128));
                     NUMERIC_OP(f32x4_sqrt, (v128), (v128));
                     NUMERIC_OP(f32x4_add, (v128, v128), (v128));
                     NUMERIC_OP(f32x4_sub, (v128, v128), (v128));
                     NUMERIC_OP(f32x4_mul, (v128, v128), (v128));
                     NUMERIC_OP(f32x4_div, (v128, v128), (v128));
                     NUMERIC_OP(f32x4_min, (v128, v128), (v128));
                     NUMERIC_OP(f32x4_max, (v128, v128), (v128));
                     NUMERIC_OP(f32x4_pmin, (v128, v128), (v128));
                     NUMERIC_OP(f32x4_pmax, (v128, v128), (v128));

                     NUMERIC_OP(f64x2_ceil, (v128), (v128));
                     NUMERIC_OP(f64x2_floor, (v128), (v128));
                     NUMERIC_OP(f64x2_trunc, (v128), (v128));
                     NUMERIC_OP(f64x2_nearest, (v128), (v128));
                     NUMERIC_OP(f64x2_abs, (v128), (v128));
                     NUMERIC_OP(f64x2_neg, (v128), (v128));
                     NUMERIC_OP(f64x2_sqrt, (v128), (v128));
                     NUMERIC_OP(f64x2_add, (v128, v128), (v128));
                     NUMERIC_OP(f64x2_sub, (v128, v128), (v128));
                     NUMERIC_OP(f64x2_mul, (v128, v128), (v128));
                     NUMERIC_OP(f64x2_div, (v128, v128), (v128));
                     NUMERIC_OP(f64x2_min, (v128, v128), (v128));
                     NUMERIC_OP(f64x2_max, (v128, v128), (v128));
                     NUMERIC_OP(f64x2_pmin, (v128, v128), (v128));
                     NUMERIC_OP(f64x2_pmax, (v128, v128), (v128));

                     NUMERIC_OP(i32x4_trunc_sat_f32x4_s, (v128), (v128));
                     NUMERIC_OP(i32x4_trunc_sat_f32x4_u, (v128), (v128));
                     NUMERIC_OP(f32x4_convert_i32x4_s, (v128), (v128));
                     NUMERIC_OP(f32x4_convert_i32x4_u, (v128), (v128));
                     NUMERIC_OP(i32x4_trunc_sat_f64x2_s_zero, (v128), (v128));
                     NUMERIC_OP(i32x4_trunc_sat_f64x2_u_zero, (v128), (v128));
                     NUMERIC_OP(f64x2_convert_low_i32x4_s, (v128), (v128));
                     NUMERIC_OP(f64x2_convert_low_i32x4_u, (v128), (v128));
                     NUMERIC_OP(f32x4_demote_f64x2_zero, (v128), (v128));
                     NUMERIC_OP(f64x2_promote_low_f32x4, (v128), (v128));

#undef NUMERIC_OP
#undef VA_SZ
#undef VA_SZ_I
#undef VA_SZ_II
#undef VA_SZ_EMPTY
#undef CAT
#undef CAT_I
#undef OUTPUTS_1
#undef OUTPUTS_0
#undef INPUTS_2
#undef INPUTS_1
#undef INPUTS_0

                     default: EOS_VM_ASSERT(false, wasm_parse_exception, "Illegal instruction");
                  }
               } break;
               case opcodes::ext_prefix: {
                  switch(parse_varuint32(code))
                  {
#define TRUNC_SAT_OP(dest, src, sign)                                                   \
                     case ext_opcodes::dest ## _trunc_sat_ ## src ## _ ## sign: {       \
                        check_in_bounds();                                              \
                        EOS_VM_ASSERT(detail::get_enable_nontrapping_fptoint(_options), wasm_parse_exception, "Non-trapping float-to-int conversions not enabled");\
                        op_stack.pop(src);                                              \
                        op_stack.push(dest);                                            \
                        code_writer.emit_ ## dest ## _trunc_sat_ ## src ## _ ## sign(); \
                     } break;
                     TRUNC_SAT_OP(i32, f32, s)
                     TRUNC_SAT_OP(i32, f32, u)
                     TRUNC_SAT_OP(i32, f64, s)
                     TRUNC_SAT_OP(i32, f64, u)
                     TRUNC_SAT_OP(i64, f32, s)
                     TRUNC_SAT_OP(i64, f32, u)
                     TRUNC_SAT_OP(i64, f64, s)
                     TRUNC_SAT_OP(i64, f64, u)
#undef TRUNC_SAT_OP
                     case ext_opcodes::memory_init: {
                        check_in_bounds();
                        EOS_VM_ASSERT(detail::get_enable_bulk_memory(_options), wasm_parse_exception, "Bulk memory not enabled");
                        EOS_VM_ASSERT(_mod->memories.size() != 0, wasm_parse_exception, "memory.init requires memory");
                        op_stack.pop(types::i32);
                        op_stack.pop(types::i32);
                        op_stack.pop(types::i32);
                        auto x = parse_varuint32(code);
                        EOS_VM_ASSERT(!!_datacount, wasm_parse_exception, "memory.init requires datacount section");
                        EOS_VM_ASSERT(x < *_datacount, wasm_parse_exception, "data segment does not exist");
                        EOS_VM_ASSERT(*code == 0, wasm_parse_exception, "memory.init must end with 0x00");
                        code++;
                        code_writer.emit_memory_init(x);
                     } break;
                     case ext_opcodes::data_drop: {
                        check_in_bounds();
                        EOS_VM_ASSERT(detail::get_enable_bulk_memory(_options), wasm_parse_exception, "Bulk memory not enabled");
                        auto x = parse_varuint32(code);
                        EOS_VM_ASSERT(!!_datacount, wasm_parse_exception, "data.drop requires datacount section");
                        EOS_VM_ASSERT(x < *_datacount, wasm_parse_exception, "data segment does not exist");
                        code_writer.emit_data_drop(x);
                     } break;
                     case ext_opcodes::memory_copy:
                        check_in_bounds();
                        EOS_VM_ASSERT(detail::get_enable_bulk_memory(_options), wasm_parse_exception, "Bulk memory not enabled");
                        EOS_VM_ASSERT(_mod->memories.size() != 0, wasm_parse_exception, "memory.copy requires memory");
                        op_stack.pop(types::i32);
                        op_stack.pop(types::i32);
                        op_stack.pop(types::i32);
                        EOS_VM_ASSERT(*code == 0, wasm_parse_exception, "memory.copy must end with 0x00 0x00");
                        code++;
                        EOS_VM_ASSERT(*code == 0, wasm_parse_exception, "memory.copy must end with 0x00 0x00");
                        code++;
                        code_writer.emit_memory_copy();
                        break;
                     case ext_opcodes::memory_fill:
                        check_in_bounds();
                        EOS_VM_ASSERT(detail::get_enable_bulk_memory(_options), wasm_parse_exception, "Bulk memory not enabled");
                        EOS_VM_ASSERT(_mod->memories.size() != 0, wasm_parse_exception, "memory.fill requires memory");
                        op_stack.pop(types::i32);
                        op_stack.pop(types::i32);
                        op_stack.pop(types::i32);
                        EOS_VM_ASSERT(*code == 0, wasm_parse_exception, "memory.fill must end with 0x00");
                        code++;
                        code_writer.emit_memory_fill();
                        break;
                     case ext_opcodes::table_init: {
                        check_in_bounds();
                        EOS_VM_ASSERT(detail::get_enable_bulk_memory(_options), wasm_parse_exception, "Bulk memory not enabled");
                        EOS_VM_ASSERT(_mod->tables.size() != 0, wasm_parse_exception, "table.init requires table");
                        auto x = parse_varuint32(code);
                        EOS_VM_ASSERT(*code == 0, wasm_parse_exception, "table.init must end with 0x00");
                        ++code;
                        op_stack.pop(types::i32);
                        op_stack.pop(types::i32);
                        op_stack.pop(types::i32);
                        code_writer.emit_table_init(x);
                     } break;
                     case ext_opcodes::elem_drop: {
                        check_in_bounds();
                        EOS_VM_ASSERT(detail::get_enable_bulk_memory(_options), wasm_parse_exception, "Bulk memory not enabled");
                        auto x = parse_varuint32(code);
                        EOS_VM_ASSERT(x < _mod->elements.size(), wasm_parse_exception, "elem segment does not exist");
                        code_writer.emit_elem_drop(x);
                     } break;
                     case ext_opcodes::table_copy:
                        check_in_bounds();
                        EOS_VM_ASSERT(detail::get_enable_bulk_memory(_options), wasm_parse_exception, "Bulk memory not enabled");
                        EOS_VM_ASSERT(_mod->tables.size() != 0, wasm_parse_exception, "table.copy requires table");
                        EOS_VM_ASSERT(*code == 0, wasm_parse_exception, "table.copy must end with 0x00 0x00");
                        ++code;
                        EOS_VM_ASSERT(*code == 0, wasm_parse_exception, "table.copy must end with 0x00 0x00");
                        ++code;
                        op_stack.pop(types::i32);
                        op_stack.pop(types::i32);
                        op_stack.pop(types::i32);
                        code_writer.emit_table_copy();
                        break;
                     default: EOS_VM_ASSERT(false, wasm_parse_exception, "Illegal instruction");
                  }
               } break;
               default: EOS_VM_ASSERT(false, wasm_parse_exception, "Illegal instruction");
            }
         }
         if constexpr (detail::has_max_stack_bytes<Options>)
         {
            code_writer.set_stack_usage(op_stack.stack_usage._max);
         }
         EOS_VM_ASSERT( pc_stack.empty(), wasm_parse_exception, "function body too long" );
         _mod->maximum_stack = std::max(_mod->maximum_stack, static_cast<uint64_t>(op_stack.maximum_operand_depth) + local_types.locals_count());
      }

      void parse_data_segment(wasm_code_ptr& code, data_segment& ds) {
         ds.index = parse_varuint32(code);
         if (ds.index == 0 || !detail::get_enable_bulk_memory(_options))
         {
            ds.passive = false;
            parse_init_expr(code, ds.offset, types::i32);
            EOS_VM_ASSERT(_mod->memories.size() != 0, wasm_parse_exception, "data requires memory");
         }
         else if (ds.index == 1)
         {
            ds.passive = true;
            ds.offset = {.opcode = opcodes::i32_const, .value = {.i32 = 0}};
         }
         else if (ds.index == 2)
         {
            ds.passive = false;
            ds.index = parse_varuint32(code);
            parse_init_expr(code, ds.offset, types::i32);
            EOS_VM_ASSERT(ds.index < _mod->memories.size(), wasm_parse_exception, "Data uses nonexistent memory");
         }
         else
         {
            EOS_VM_ASSERT(false, wasm_parse_exception, "Unexpected flag for data");
         }
         auto len =  parse_varuint32(code);
         EOS_VM_ASSERT(len <= detail::get_max_data_segment_bytes(_options), wasm_parse_exception, "data segment too large.");
         EOS_VM_ASSERT(static_cast<uint64_t>(static_cast<uint32_t>(ds.offset.value.i32)) + len <= detail::get_max_linear_memory_init(_options),
                       wasm_parse_exception, "out-of-bounds data section");
         auto guard = code.scoped_shrink_bounds(len);
         ds.data.assign(code.raw(), code.raw() + len);
         code += len;
      }

      template <typename Elem, typename ParseFunc>
      inline void parse_section_impl(wasm_code_ptr& code, vec<Elem>& elems, std::uint32_t max_elements, ParseFunc&& elem_parse) {
         auto count = parse_varuint32(code);
         EOS_VM_ASSERT(count <= max_elements, wasm_parse_exception, "number of section elements exceeded limit");
         elems      = vec<Elem>{ _allocator, count };
         for (size_t i = 0; i < count; i++) { elem_parse(code, elems.at(i), i); }
      }

      template <typename Elem, typename ParseFunc>
      inline void parse_section_impl(wasm_code_ptr& code, std::vector<Elem>& elems, std::uint32_t max_elements, ParseFunc&& elem_parse) {
         auto count = parse_varuint32(code);
         EOS_VM_ASSERT(count <= max_elements, wasm_parse_exception, "number of section elements exceeded limit");
         elems.resize(count);
         for (size_t i = 0; i < count; i++) { elem_parse(code, elems.at(i), i); }
      }

      template <uint8_t id>
      requires (id == section_id::type_section)
      inline void parse_section(wasm_code_ptr&          code,
                                std::vector<func_type>& elems) {
         parse_section_impl(code, elems, detail::get_max_type_section_elements(_options),
                            [&](wasm_code_ptr& code, func_type& ft, std::size_t /*idx*/) { parse_func_type(code, ft); });
      }
      template <uint8_t id>
      requires (id == section_id::import_section)
      inline void parse_section(wasm_code_ptr&             code,
                                std::vector<import_entry>& elems) {
         parse_section_impl(code, elems, detail::get_max_import_section_elements(_options),
                            [&](wasm_code_ptr& code, import_entry& ie, std::size_t /*idx*/) { parse_import_entry(code, ie); });
      }
      template <uint8_t id>
      requires (id == section_id::function_section)
      inline void parse_section(wasm_code_ptr&         code,
                                std::vector<uint32_t>& elems) {
         parse_section_impl(code, elems, detail::get_max_function_section_elements(_options),
                            [&](wasm_code_ptr& code, uint32_t& elem, std::size_t /*idx*/) { elem = parse_varuint32(code); });
      }
      template <uint8_t id>
      requires (id == section_id::table_section)
      inline void parse_section(wasm_code_ptr&           code,
                                std::vector<table_type>& elems) {
         parse_section_impl(code, elems, 1,
                            [&](wasm_code_ptr& code, table_type& tt, std::size_t /*idx*/) { parse_table_type(code, tt); });
      }
      template <uint8_t id>
      requires (id == section_id::memory_section)
      inline void parse_section(wasm_code_ptr&            code,
                                std::vector<memory_type>& elems) {
         parse_section_impl(code, elems, 1, [&](wasm_code_ptr& code, memory_type& mt, std::size_t idx) {
            EOS_VM_ASSERT(idx == 0, wasm_parse_exception, "only one memory is permitted");
            parse_memory_type(code, mt);
         });
      }
      template <uint8_t id>
      requires (id == section_id::global_section)
      inline void parse_section(wasm_code_ptr&                code,
                                std::vector<global_variable>& elems) {
         parse_section_impl(code, elems, detail::get_max_global_section_elements(_options),
                            [&](wasm_code_ptr& code, global_variable& gv, std::size_t /*idx*/) { parse_global_variable(code, gv); });
      }
      template <uint8_t id>
      requires (id == section_id::export_section)
      inline void parse_section(wasm_code_ptr&             code,
                                std::vector<export_entry>& elems) {
         parse_section_impl(code, elems, detail::get_max_export_section_elements(_options),
                            [&](wasm_code_ptr& code, export_entry& ee, std::size_t /*idx*/) { parse_export_entry(code, ee); });
      }
      template <uint8_t id>
      inline void parse_section(wasm_code_ptr&                                                        code,
                                typename std::enable_if_t<id == section_id::start_section, uint32_t>& start) {
         start = parse_varuint32(code);
         const func_type& ft = _mod->get_function_type(start);
         EOS_VM_ASSERT(ft.return_count == 0 && ft.param_types.size() == 0, wasm_parse_exception, "wrong type for start");
      }
      template <uint8_t id>
      requires (id == section_id::element_section)
      inline void parse_section(wasm_code_ptr&             code,
                                std::vector<elem_segment>& elems) {
         parse_section_impl(code, elems, detail::get_max_element_section_elements(_options),
                            [&](wasm_code_ptr& code, elem_segment& es, std::size_t /*idx*/) { parse_elem_segment(code, es); });
      }
      template <uint8_t id>
      requires (id == section_id::code_section)
      inline void parse_section(wasm_code_ptr&              code,
                                std::vector<function_body>& elems) {
         const void* code_start = code.raw() - code.offset();
         parse_section_impl(code, elems, detail::get_max_function_section_elements(_options),
                            [&](wasm_code_ptr& code, function_body& fb, std::size_t idx) { parse_function_body(code, fb, idx); });
         EOS_VM_ASSERT( elems.size() == _mod->functions.size(), wasm_parse_exception, "code section must have the same size as the function section" );

         write_code_out(_allocator, code, code_start);
      }

      void write_code_out(growable_allocator& allocator, wasm_code_ptr& code, const void* code_start) {
         Writer code_writer(allocator, code.bounds() - code.offset(), *_mod);
         imap.on_code_start(code_writer.get_base_addr(), code_start);
         for (size_t i = 0; i < _function_bodies.size(); i++) {
            function_body& fb = _mod->code[i];
            func_type& ft = _mod->types.at(_mod->functions.at(i));
            local_types_t local_types(ft, fb.locals);
            imap.on_function_start(code_writer.get_addr(), _function_bodies[i].first.raw());
            code_writer.emit_prologue(ft, fb.locals, i);
            parse_function_body_code(_function_bodies[i].first, fb.size, _function_bodies[i].second, code_writer, ft, local_types);
            code_writer.emit_epilogue(ft, fb.locals, i);
            imap.on_function_end(code_writer.get_addr(), _function_bodies[i].first.bnds);
            code_writer.finalize(fb);
         }
         imap.on_code_end(code_writer.get_addr(), code.raw());
      }

      template <uint8_t id>
      requires (id == section_id::data_section)
      inline void parse_section(wasm_code_ptr&             code,
                                std::vector<data_segment>& elems) {
         parse_section_impl(code, elems, detail::get_max_data_section_elements(_options),
                            [&](wasm_code_ptr& code, data_segment& ds, std::size_t /*idx*/) { parse_data_segment(code, ds); });
         if (_datacount) {
            EOS_VM_ASSERT(*_datacount == elems.size(), wasm_parse_exception, "data count does not match data");
         }
      }
      template <uint8_t id>
      requires (id == section_id::data_count_section)
      inline void parse_section(wasm_code_ptr& code, std::optional<std::uint32_t>& n)
      {
         n = parse_varuint32(code);
      }

      template <size_t N>
      varint<N> parse_varint(const wasm_code& code, size_t index) {
         varint<N> result(0);
         result.set(code, index);
         return result;
      }

      template <size_t N>
      varuint<N> parse_varuint(const wasm_code& code, size_t index) {
         varuint<N> result(0);
         result.set(code, index);
         return result;
      }

      void on_mutable_global(uint8_t type) {
         _globals_checker.on_mutable_global(_options, type);
      }

      void validate_exports() const {
         std::vector<const std::vector<uint8_t>*> export_names;
         export_names.reserve(_mod->exports.size());
         for (uint32_t i = 0; i < _mod->exports.size(); ++i) {
            export_names.push_back(&_mod->exports[i].field_str);
         }
         std::sort(export_names.begin(), export_names.end(), [](auto* lhs, auto* rhs) {
            return *lhs < *rhs;
         });
         auto it = std::adjacent_find(export_names.begin(), export_names.end(), [](auto* lhs, auto* rhs) {
            return *lhs == *rhs;
         });
         EOS_VM_ASSERT(it == export_names.end(), wasm_parse_exception, "duplicate export name");
      }

    private:
      growable_allocator& _allocator;
      Options             _options;
      module*             _mod; // non-owning weak pointer
      int64_t             _current_function_index = -1;
      uint64_t            _maximum_function_stack_usage = 0; // non-parameter locals + stack
      std::vector<std::pair<wasm_code_ptr, detail::max_func_local_bytes_stack_checker<Options>>>  _function_bodies;
      detail::max_mutable_globals_checker<Options> _globals_checker;
      detail::eosio_max_nested_structures_checker<Options> _nested_checker;
      std::optional<std::uint32_t> _datacount;
      typename DebugInfo::builder imap;
      std::vector<uint32_t> type_aliases;
      std::vector<uint32_t> fast_functions;
   };
}} // namespace eosio::vm
