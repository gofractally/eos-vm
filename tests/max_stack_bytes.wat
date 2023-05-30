(module
   (func $host.call (import "env" "host.call"))
   (func (export "empty")) ;; should succeed iff the stack limit is >= 16
   (func $i32.stack (export "i32.stack") (param i32)
      (if (local.get 0) (then
         (local.get 0)
         (i32.const -1)
         (i32.add)
         (call $i32.stack)
      ))
   )
)
