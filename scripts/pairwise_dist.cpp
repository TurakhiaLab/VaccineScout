// For every sequence in a parquet dictionary (columns: key = Sample_ID,
// value = aligned AA sequence), compute the Hamming distance against every
// sequence including itself (i == j gives distance 0) and write a parquet
// with columns Seq_1, Seq_2, Distance.
//
// Distances are computed in parallel into a pre-allocated N*N int32 matrix
// and then streamed to the output file in strict outer-i order, so the
// resulting parquet contains rows in the order
//     (i=0, j=0..N-1), (i=1, j=0..N-1), ..., (i=N-1, j=0..N-1)
// regardless of TBB scheduling. Extra memory: N*N*4 bytes for the matrix.

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>
#include <parquet/column_reader.h>
#include <parquet/file_reader.h>
#include <parquet/types.h>

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

static void read_byte_array_column(parquet::RowGroupReader* rg,
                                   const parquet::ColumnDescriptor& descr, int col_index,
                                   std::vector<std::string>* out) {
  std::shared_ptr<parquet::ColumnReader> col_reader = rg->Column(col_index);
  auto* ba_reader = dynamic_cast<parquet::ByteArrayReader*>(col_reader.get());
  if (!ba_reader)
    throw std::runtime_error("internal: column " + std::to_string(col_index) +
                             " is not a BYTE_ARRAY reader");
  if (descr.physical_type() != parquet::Type::BYTE_ARRAY)
    throw std::runtime_error("internal: column type mismatch");

  const int16_t max_def = descr.max_definition_level();
  const int16_t max_rep = descr.max_repetition_level();
  int16_t* rep_levels = nullptr;
  std::vector<int16_t> rep_storage;
  constexpr int64_t k_batch = 65536;
  if (max_rep > 0) {
    rep_storage.resize(static_cast<std::size_t>(k_batch));
    rep_levels = rep_storage.data();
  }
  std::vector<int16_t> def_levels(static_cast<std::size_t>(k_batch));
  std::vector<parquet::ByteArray> values(static_cast<std::size_t>(k_batch));

  for (;;) {
    int64_t values_read = 0;
    const int64_t levels_read =
        ba_reader->ReadBatch(k_batch, def_levels.data(), rep_levels, values.data(), &values_read);
    if (levels_read <= 0)
      break;
    int64_t vi = 0;
    for (int64_t i = 0; i < levels_read; ++i) {
      if (max_def == 0 || def_levels[static_cast<std::size_t>(i)] == max_def) {
        if (vi >= values_read)
          throw std::runtime_error("parquet ReadBatch internal value index overrun");
        const parquet::ByteArray& ba = values[static_cast<std::size_t>(vi++)];
        out->emplace_back(reinterpret_cast<const char*>(ba.ptr), static_cast<std::size_t>(ba.len));
      }
    }
    if (vi != values_read)
      throw std::runtime_error("parquet ReadBatch definition levels vs values mismatch");
  }
}

static void load_sequence_dict_vectors(const fs::path& parquet_path,
                                       std::vector<std::string>* out_ids,
                                       std::vector<std::string>* out_seqs) {
  auto in_res = arrow::io::ReadableFile::Open(parquet_path.string());
  if (!in_res.ok())
    throw std::runtime_error(in_res.status().ToString());
  std::shared_ptr<arrow::io::ReadableFile> infile = in_res.ValueOrDie();

  std::unique_ptr<parquet::ParquetFileReader> pq_reader = parquet::ParquetFileReader::Open(infile);
  std::shared_ptr<parquet::FileMetaData> meta = pq_reader->metadata();
  const parquet::SchemaDescriptor* pschema = meta->schema();
  if (pschema->num_columns() != 2)
    throw std::runtime_error(
        "parquet must have exactly 2 columns (key, value); file has " +
        std::to_string(pschema->num_columns()));
  const parquet::ColumnDescriptor* pkey = pschema->Column(0);
  const parquet::ColumnDescriptor* pval = pschema->Column(1);
  if (pkey->name() != "key" || pval->name() != "value")
    throw std::runtime_error(
        "parquet columns must be named 'key' and 'value' in order; got '" + pkey->name() +
        "', '" + pval->name() + "'");
  if (pkey->physical_type() != parquet::Type::BYTE_ARRAY ||
      pval->physical_type() != parquet::Type::BYTE_ARRAY)
    throw std::runtime_error("parquet key/value must be BYTE_ARRAY (UTF-8 strings)");

  const int64_t nrows = meta->num_rows();
  if (nrows <= 0)
    throw std::runtime_error("empty parquet table: " + parquet_path.string());

  out_ids->reserve(static_cast<std::size_t>(nrows));
  out_seqs->reserve(static_cast<std::size_t>(nrows));

  const int n_rg = meta->num_row_groups();
  std::size_t loaded = 0;
  for (int rg = 0; rg < n_rg; ++rg) {
    std::shared_ptr<parquet::RowGroupReader> rgr = pq_reader->RowGroup(rg);
    std::vector<std::string> keys;
    std::vector<std::string> vals;
    read_byte_array_column(rgr.get(), *pkey, 0, &keys);
    read_byte_array_column(rgr.get(), *pval, 1, &vals);
    if (keys.size() != vals.size())
      throw std::runtime_error("parquet row group " + std::to_string(rg) +
                               ": key/value row count mismatch");
    for (std::size_t i = 0; i < keys.size(); ++i) {
      out_ids->push_back(std::move(keys[i]));
      out_seqs->push_back(std::move(vals[i]));
    }
    loaded += keys.size();
  }
  if (static_cast<int64_t>(loaded) != nrows)
    throw std::runtime_error("parquet rows read (" + std::to_string(loaded) +
                             ") != footer num_rows (" + std::to_string(nrows) + ")");
}

struct PairBuffer {
  std::vector<std::string> s1;
  std::vector<std::string> s2;
  std::vector<int32_t> d;

  void clear() {
    s1.clear();
    s2.clear();
    d.clear();
  }
  std::size_t size() const {
    return d.size();
  }
};

static void flush_buffer(parquet::arrow::FileWriter* writer,
                         const std::shared_ptr<arrow::Schema>& schema,
                         PairBuffer& buf) {
  if (buf.size() == 0)
    return;

  auto check = [](const arrow::Status& s, const char* msg) {
    if (!s.ok())
      throw std::runtime_error(std::string(msg) + ": " + s.ToString());
  };

  arrow::StringBuilder b1;
  arrow::StringBuilder b2;
  arrow::Int32Builder bd;
  const int64_t n = static_cast<int64_t>(buf.size());
  check(b1.Reserve(n), "reserve Seq_1");
  check(b2.Reserve(n), "reserve Seq_2");
  check(bd.Reserve(n), "reserve Distance");
  for (const auto& s : buf.s1)
    check(b1.Append(s), "append Seq_1");
  for (const auto& s : buf.s2)
    check(b2.Append(s), "append Seq_2");
  for (int32_t v : buf.d)
    check(bd.Append(v), "append Distance");

  std::shared_ptr<arrow::Array> a1;
  std::shared_ptr<arrow::Array> a2;
  std::shared_ptr<arrow::Array> ad;
  check(b1.Finish(&a1), "finish Seq_1");
  check(b2.Finish(&a2), "finish Seq_2");
  check(bd.Finish(&ad), "finish Distance");

  std::shared_ptr<arrow::Table> table = arrow::Table::Make(schema, {a1, a2, ad});
  arrow::Status s = writer->WriteTable(*table, table->num_rows());
  if (!s.ok())
    throw std::runtime_error("write parquet: " + s.ToString());

  buf.clear();
}

static void usage(const char* prog) {
  std::cerr
      << "Usage: " << prog
      << " --parquet <sequence_dict.parquet> --output <out.parquet>\n"
         "\n"
         "Reads a parquet with two BYTE_ARRAY columns named 'key' (Sample_ID) and\n"
         "'value' (aligned AA sequence). For every sequence i, computes the Hamming\n"
         "distance to every sequence j (including j == i, distance 0). Sequences of\n"
         "unequal length still error on off-diagonal pairs. Writes a parquet with\n"
         "columns Seq_1, Seq_2, Distance (N*N rows, streamed in row groups).\n";
}

int main(int argc, char** argv) {
  fs::path parquet_path;
  fs::path output_path;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--parquet" && i + 1 < argc) {
      parquet_path = argv[++i];
    } else if (a == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    } else {
      std::cerr << "unknown argument: " << a << "\n";
      usage(argv[0]);
      return 1;
    }
  }
  if (parquet_path.empty() || output_path.empty()) {
    usage(argv[0]);
    return 1;
  }

  try {
    std::vector<std::string> ids;
    std::vector<std::string> seqs;
    load_sequence_dict_vectors(parquet_path, &ids, &seqs);

    const std::size_t n = ids.size();
    if (n == 0)
      throw std::runtime_error("need at least 1 sequence to compute pairwise distances");

    unsigned thread_cap = std::thread::hardware_concurrency();
    if (thread_cap == 0)
      thread_cap = 1;
    tbb::global_control tbb_threads(tbb::global_control::max_allowed_parallelism,
                                    static_cast<std::size_t>(thread_cap));

    auto schema = arrow::schema({
        arrow::field("Seq_1", arrow::utf8()),
        arrow::field("Seq_2", arrow::utf8()),
        arrow::field("Distance", arrow::int32()),
    });

    auto out_res = arrow::io::FileOutputStream::Open(output_path.string());
    if (!out_res.ok())
      throw std::runtime_error("open output: " + out_res.status().ToString());
    std::shared_ptr<arrow::io::FileOutputStream> out_stream = out_res.ValueOrDie();

    std::shared_ptr<parquet::WriterProperties> writer_props =
        parquet::WriterProperties::Builder().compression(parquet::Compression::SNAPPY)->build();
    std::shared_ptr<parquet::ArrowWriterProperties> arrow_props =
        parquet::ArrowWriterProperties::Builder().build();

    auto writer_res = parquet::arrow::FileWriter::Open(
        *schema, arrow::default_memory_pool(), out_stream, writer_props, arrow_props);
    if (!writer_res.ok())
      throw std::runtime_error("open file writer: " + writer_res.status().ToString());
    std::unique_ptr<parquet::arrow::FileWriter> writer = std::move(writer_res).ValueOrDie();

    constexpr std::size_t kFlushThreshold = 1u << 20;  // ~1M rows per row group

    const std::size_t total_pairs = n * n;
    if (n != 0 && total_pairs / n != n)
      throw std::runtime_error(
          "N*N overflows std::size_t (N = " + std::to_string(n) + ")");

    std::vector<int32_t> dist(total_pairs, 0);

    std::atomic<std::size_t> progress{0};
    std::mutex log_mu;
    const std::size_t report_step = std::max<std::size_t>(1u, n / 100u);

    // Compute all distances in parallel; row i is written into dist[i*n .. i*n+n).
    // The diagonal (j == i) stays at the initial 0.
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, n),
        [&](const tbb::blocked_range<std::size_t>& r) {
          for (std::size_t i = r.begin(); i < r.end(); ++i) {
            const std::string& a = seqs[i];
            const std::size_t alen = a.size();
            const std::string& aid = ids[i];
            int32_t* row = dist.data() + i * n;

            for (std::size_t j = 0; j < n; ++j) {
              if (i == j)
                continue;
              const std::string& b = seqs[j];
              if (b.size() != alen)
                throw std::runtime_error(
                    "sequence length mismatch between " + aid + " and " + ids[j] +
                    " (" + std::to_string(alen) + " vs " + std::to_string(b.size()) + ")");
              int d = 0;
              const char* pa = a.data();
              const char* pb = b.data();
              for (std::size_t k = 0; k < alen; ++k) {
                if (pa[k] != pb[k])
                  ++d;
              }
              row[j] = static_cast<int32_t>(d);
            }

            const std::size_t done = progress.fetch_add(1, std::memory_order_relaxed) + 1;
            if (done == n || done % report_step == 0) {
              std::lock_guard<std::mutex> lk(log_mu);
              std::cerr << "\rpairwise_dist (compute) " << done << " / " << n << std::flush;
            }
          }
        });
    std::cerr << "\n";

    // Sequentially stream rows out in strict i order.
    PairBuffer buf;
    const std::size_t reserve_rows = std::min<std::size_t>(kFlushThreshold + n, total_pairs);
    buf.s1.reserve(reserve_rows);
    buf.s2.reserve(reserve_rows);
    buf.d.reserve(reserve_rows);

    std::size_t written = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const std::string& aid = ids[i];
      const int32_t* row = dist.data() + i * n;
      for (std::size_t j = 0; j < n; ++j) {
        buf.s1.push_back(aid);
        buf.s2.push_back(ids[j]);
        buf.d.push_back(row[j]);
      }
      if (buf.size() >= kFlushThreshold) {
        flush_buffer(writer.get(), schema, buf);
      }
      written += n;
      if (i + 1 == n || (i + 1) % report_step == 0) {
        std::cerr << "\rpairwise_dist (write)   " << (i + 1) << " / " << n
                  << " (" << written << " rows)" << std::flush;
      }
    }
    if (buf.size() > 0)
      flush_buffer(writer.get(), schema, buf);
    std::cerr << "\n";

    arrow::Status close_status = writer->Close();
    if (!close_status.ok())
      throw std::runtime_error("close parquet writer: " + close_status.ToString());
    arrow::Status close_stream = out_stream->Close();
    if (!close_stream.ok())
      throw std::runtime_error("close output: " + close_stream.ToString());

    std::cout << "wrote " << output_path << " (" << total_pairs << " rows)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "pairwise_dist: " << e.what() << '\n';
    return 1;
  }
}
