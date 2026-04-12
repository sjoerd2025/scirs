# scirs2-io TODO

## v0.3.3 Completed

### Classic Scientific Formats
- [x] MATLAB `.mat` v4/v5 read/write with all data types, structures, cell arrays
- [x] WAV audio read/write
- [x] ARFF (Attribute-Relation File Format) read/write
- [x] NetCDF3 and NetCDF4/HDF5 with unlimited dimensions and chunking
- [x] HDF5-lite pure-Rust hierarchical data reader
- [x] Matrix Market and Harwell-Boeing sparse matrix formats

### Modern Columnar and Binary Formats
- [x] Parquet-lite: pure-Rust Parquet reader
- [x] Feather (Arrow IPC): memory-mapped columnar format
- [x] ORC format reader
- [x] Binary format encoding utilities

### Serialization Formats
- [x] CBOR (RFC 7049) serialization and deserialization
- [x] BSON (Binary JSON) encode/decode
- [x] Avro schema-based serialization with schema evolution
- [x] Protobuf-lite: pure-Rust protobuf encoding/decoding
- [x] MessagePack encode/decode
- [x] NDJSON (Newline-Delimited JSON) streaming reader

### Streaming and Lazy Evaluation
- [x] Streaming CSV with lazy chunk evaluation
- [x] Streaming JSON incremental parser
- [x] NDJSON line-by-line streaming
- [x] Arrow IPC framed streaming
- [x] Backpressure-aware pipeline (sources, transforms, sinks)
- [x] Typed transform pipeline

### Compression
- [x] LZ4 high-speed compression
- [x] Zstd compression with configurable levels
- [x] Brotli general-purpose compression
- [x] Snappy block compression
- [x] GZIP / BZIP2 deflate-based compression
- [x] Parallel chunk compression (up to 2.5x throughput)

### Data Catalog, Lineage, Governance
- [x] Data catalog: register, tag, discover datasets
- [x] Lineage tracking: record transformations and provenance
- [x] Schema registry: store, evolve, and validate schemas
- [x] Dataset versioning with diff and rollback

### ETL and Query
- [x] ETL pipeline framework: source -> transform -> sink with parallel stages
- [x] SQL-like query interface: predicate pushdown and projection
- [x] Universal reader: auto-detect format from magic bytes/extension
- [x] Format detection for dozens of formats

### Cloud and Distributed
- [x] Cloud storage connector framework (AWS S3, GCS, Azure Blob)
- [x] Distributed / partitioned parallel read/write

### Validation and Integrity
- [x] CRC32, SHA-256, BLAKE3 checksum verification
- [x] JSON Schema-compatible schema validation engine
- [x] Format-specific structural validators

## v0.4.0 Roadmap

### New Formats
- [x] Zarr v2/v3 format: chunked, compressed, N-dimensional arrays; compatible with Zarr-Python — Implemented in v0.4.0 (`zarr/` module)
- [x] TileDB integration: dense and sparse multi-dimensional arrays for analytics — Implemented in v0.4.0 (`tiledb.rs` module)
- [x] Lance format: modern columnar format for ML datasets — Implemented in v0.4.0 (`lance/` module)
- [x] Delta Lake log-based table format reader — Implemented in v0.4.0 (`delta.rs` module)
- [x] Iceberg table format support — implemented in v0.4.2 (`iceberg.rs`)

### Transport Protocols
- [x] Apache Arrow Flight protocol: high-throughput gRPC-based data transfer — Implemented in v0.4.0 (`protocols/arrow_flight.rs`)
- [x] Apache Kafka consumer/producer for streaming scientific data — Implemented in v0.4.0 (`protocols/kafka.rs`)
- [x] MQTT topic-based streaming for IoT/sensor data ingestion — Implemented in v0.4.0 (`mqtt_broker/` module)

### Compression and Encoding
- [x] Columnar-aware compression: dictionary encoding, RLE, delta encoding per column — Implemented in v0.4.0 (`columnar/dictionary.rs`, `columnar/rle.rs`, `columnar/delta.rs`)
- [x] Bloom filter indexes for Parquet-like predicate pushdown — Implemented in v0.4.0 (`analytics/bloom_index.rs`)
- [x] FSST (Fast Static Symbol Table) string compression — Implemented in v0.4.0 (`columnar/fsst.rs`)
- [x] Adaptive compression: auto-select algorithm based on data entropy — Implemented in v0.4.2 (`adaptive_compression/mod.rs`); OxiARC-backed LZ4/Zstd/Brotli with Shannon entropy selection; `auto_compress`/`auto_decompress` with 1-byte tag

### Cloud and Distributed
- [x] Native AWS S3 multipart upload with parallel chunk upload — Implemented in v0.4.2 (`s3_multipart.rs`); feature-gated stub with full state machine simulation; real HTTP requires `aws-sdk-s3` feature
- [x] Native GCS resumable uploads — Implemented in v0.4.2 (`cloud/gcs.rs`); simulation-mode state machine with offset validation, abort/finalize, assembled_data; 8 tests
- [x] Azure Blob SAS-token authentication support — Implemented in v0.4.2 (`cloud/azure_sas.rs`); SasPermissions, SasResource, generate_sas_token, build_sas_url, parse_sas_token, is_sas_valid; 8 tests
- [x] Object-store abstraction layer unified across providers — Implemented in v0.4.2 (`cloud/mod.rs`); `ObjectStore` trait + `LocalObjectStore` + `MemoryObjectStore` + S3/GCS/Azure stubs; `parse_store_url` + `from_url` factory; GCS and Azure stubs available, feature-gated

### Query and Analytics
- [x] DataFusion-compatible table provider interface — implemented in v0.4.2 (`datafusion_provider.rs`)
- [x] Vectorized expression evaluation for filter and project — implemented in v0.4.2 (`datafusion_provider.rs`)
- [x] Approximate aggregations: HyperLogLog, t-digest, count-min sketch — Implemented in v0.4.0 (`analytics/hyperloglog.rs`, `analytics/tdigest.rs`, `analytics/count_min.rs`)
- [x] Join algorithms for cross-format dataset merge — implemented in v0.4.2 (`joins.rs`)

### Streaming Enhancements
- [x] Exactly-once delivery semantics for streaming pipeline sinks — Implemented in v0.4.2 (`exactly_once.rs`); WriteAheadLog (disk + in-memory) + ExactlyOnceSink with idempotency-key deduplication; 10 tests
- [x] Windowed aggregation (tumbling, sliding, session windows) — Implemented in v0.4.0 (`streaming/windows.rs`)
- [x] Watermark-based late-data handling — Implemented in v0.4.0 (`streaming/watermark.rs`)
- [x] Checkpointing and restart for long-running streaming jobs — Implemented in v0.4.0 (`streaming/checkpoint.rs`)

### Machine Learning Integration
- [x] Tensor serialization (safetensors-compatible read/write) — implemented in v0.4.2 (`tensors/safetensors.rs`)
- [x] ONNX model proto read/write — implemented in v0.4.2 (`tensors/onnx_proto.rs`)
- [x] TFRecord reader for TensorFlow data pipelines — implemented in v0.4.2 (`tensors/tfrecord.rs`)
- [x] Efficient mini-batch sampler with shuffle and stratified splitting — implemented in v0.4.2 (`minibatch.rs`)

## Known Issues

- Large HDF5 files with deeply nested groups may be slow on the pure-Rust hdf5-lite reader; the system-library `hdf5` feature should be preferred for those workloads.
- The ORC reader does not yet support all column encodings (RLE v2, dictionary, DIRECT_V2); unsupported columns fall back to raw bytes.
- Arrow IPC streaming does not yet validate all IPC message types; unknown message types are silently skipped.
- Cloud connector framework provides the interface only; actual HTTP signing and chunked transfer require activating the `reqwest` feature and providing credentials at runtime.
- BSON serialization of f32 arrays upcasts to f64 to conform with the BSON type system.
