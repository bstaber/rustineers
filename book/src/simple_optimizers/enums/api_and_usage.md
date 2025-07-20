## API and usage

This pattern can be a clean alternative to trait-based designs when you want:
- A small number of well-known variants
- Built-in state encapsulation
- Exhaustive handling via pattern matching

It keeps related logic grouped under one type and can be extended easily with new optimizers.