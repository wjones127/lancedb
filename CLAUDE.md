LanceDB is a database designed for retrieval, including vector, full-text, and hybrid search.
It is a wrapper around Lance. There are two backends: local (in-process like SQLite) and
remote (against LanceDB Cloud).

The core of LanceDB is written in Rust. There are bindings in Python, Typescript, and Java.

Project layout:

* `rust/lancedb`: The LanceDB core Rust implementation.
* `python`: The Python bindings, using PyO3.
* `nodejs`: The Typescript bindings, using napi-rs
* `java`: The Java bindings

(`rust/ffi` and `node/` are for a deprecated package. You can ignore them.)

Common commands:

* Check for compiler errors: `cargo check --features remote --tests --examples`
* Run tests: `cargo test --features remote --tests`
* Run specific test: `cargo test --features remote -p <package_name> --test <test_name>`
* Lint: `cargo clippy --features remote --tests --examples`
* Format: `cargo fmt --all`

Before committing changes, run formatting.

## Code Structure Example: adding a new method on Table

The base implementation is in `rust/lancedb` and has two backends: local and remote.
Local is a wrapper around a Lance Dataset, while remote is a REST API client
against LanceDB Cloud.

### Step 1: Rust Core Implementation

* **Add method signature to `BaseTable` trait** in `rust/lancedb/src/table.rs`
* **Add public method to `Table` struct** in `rust/lancedb/src/table.rs` that delegates to the trait
* **Implement for `NativeTable`** in `rust/lancedb/src/table.rs` using Lance Dataset APIs
* **Implement for `RemoteTable`** in `rust/lancedb/src/remote/table.rs`:
    * Use REST API endpoint (may need to add HTTP method to `RestfulLanceDbClient` in `rust/lancedb/src/remote/client.rs`)
    * Add unit test with mock request/response following pattern at line 2624+
    * Test should verify HTTP method, URL path, and request body serialization

### Step 2: Python Bindings (Complete Implementation Required)

* **Rust binding**: Add method in `python/src/table.rs` using PyO3:
    * Use `future_into_py()` for async methods
* **Type annotations**: Add to `python/python/lancedb/_lancedb.pyi`
* **Sync API implementations**:
    * Abstract method in base `Table` class in `python/python/lancedb/table.py`
    * Implementation in `LanceTable` class using `LOOP.run()`
    * Implementation in `RemoteTable` class in `python/python/lancedb/remote/table.py`
* **Async API**: Add to `AsyncTable` class in `python/python/lancedb/table.py`
* **Tests**: Add to `python/python/tests/test_table.py` for local implementation

### Step 3: TypeScript Bindings (Complete Implementation Required)

* **Rust binding**: Add method in `nodejs/src/table.rs` using napi-rs
* **TypeScript API**: Add to abstract `Table` class in `nodejs/lancedb/table.ts`:
    * Add abstract method with JSDoc documentation
    * Add concrete implementation that calls the Rust binding
* **Tests**: Add to `nodejs/__test__/table.test.ts` for local implementation

### Step 4: Build and Test

* **Format**: Run `cargo fmt --all`
* **Build**:
    * Node.js: `cd nodejs && npm run build`
    * Python: `cd python && make develop`
* **Test**:
    * Rust: `cargo test --features remote test_your_method_name`
    * Node.js: `npm test -- --testNamePattern="yourMethodName"`
    * Python: `python -m pytest python/tests/test_table.py::test_your_method_name -v`

### Important Notes

* **Python bindings**: The method must be inside a `#[pymethods]` impl block to be exported
* **TypeScript**: Both abstract declaration and concrete implementation are required
* **Testing**: Each backend (local/remote) and each language should have tests
* **Error handling**: Follow existing patterns for each language binding
