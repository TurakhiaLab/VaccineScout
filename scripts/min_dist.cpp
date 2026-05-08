// Minimum mutation (Hamming) distance from each future Sequence_ID to the nearest sequence in the current set. 

#include <arrow/io/api.h>
#include <parquet/column_reader.h>
#include <parquet/file_reader.h>
#include <parquet/types.h>

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <filesystem>

namespace fs = std::filesystem;

static std::string trim(std::string s) {
  while (!s.empty() &&
         (s.back() == '\r' || s.back() == '\n' || std::isspace(static_cast<unsigned char>(s.back()))))
    s.pop_back();
  std::size_t i = 0;
  while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
    ++i;
  return s.substr(i);
}

static std::string to_lower_ascii(std::string s) {
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z')
      c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
}

static std::string first_csv_field(const std::string& line) {
  std::string s = trim(line);
  if (s.empty())
    return s;
  std::size_t comma = s.find(',');
  if (comma == std::string::npos)
    return s;
  std::string field = trim(s.substr(0, comma));
  if (field.size() >= 2u && field.front() == '"' && field.back() == '"')
    field = field.substr(1, field.size() - 2u);
  return field;
}

// One column per row, optional header Sample_ID or Sequence_ID (pandas to_csv index=False).
static std::vector<std::string> read_sample_ids_csv(const fs::path& path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open: " + path.string());
  std::vector<std::string> raw;
  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#')
      continue;
    raw.push_back(std::move(line));
  }
  if (raw.empty())
    return {};
  std::size_t start = 0;
  const std::string hdr = to_lower_ascii(first_csv_field(raw[0]));
  if (hdr == "sample_id" || hdr == "sequence_id")
    start = 1;
  std::vector<std::string> out;
  out.reserve(raw.size() - start);
  for (std::size_t i = start; i < raw.size(); ++i) {
    std::string id = first_csv_field(raw[i]);
    if (!id.empty())
      out.push_back(std::move(id));
  }
  return out;
}

static int hamming_equal_length(const std::string& a, const std::string& b) {
  if (a.size() != b.size())
    return std::numeric_limits<int>::max();
  int d = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i])
      ++d;
  return d;
}

static void read_byte_array_column(parquet::RowGroupReader* rg, const parquet::ColumnDescriptor& descr,
                                   int col_index, std::vector<std::string>* out) {
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

static std::unordered_map<std::string, std::string> load_sequence_dict_parquet(const fs::path& parquet_path) {
  auto in_res = arrow::io::ReadableFile::Open(parquet_path.string());
  if (!in_res.ok())
    throw std::runtime_error(in_res.status().ToString());
  std::shared_ptr<arrow::io::ReadableFile> infile = in_res.ValueOrDie();

  std::unique_ptr<parquet::ParquetFileReader> pq_reader = parquet::ParquetFileReader::Open(infile);
  std::shared_ptr<parquet::FileMetaData> meta = pq_reader->metadata();
  const parquet::SchemaDescriptor* pschema = meta->schema();
  if (pschema->num_columns() != 2)
    throw std::runtime_error(
        "sequence_dict.parquet must have exactly 2 columns (key, value); file has " +
        std::to_string(pschema->num_columns()));
  const parquet::ColumnDescriptor* pkey = pschema->Column(0);
  const parquet::ColumnDescriptor* pval = pschema->Column(1);
  if (pkey->name() != "key" || pval->name() != "value")
    throw std::runtime_error(
        "sequence_dict.parquet columns must be named 'key' and 'value' in order; got '" +
        pkey->name() + "', '" + pval->name() + "'");
  if (pkey->physical_type() != parquet::Type::BYTE_ARRAY ||
      pval->physical_type() != parquet::Type::BYTE_ARRAY)
    throw std::runtime_error("sequence_dict.parquet key/value must be BYTE_ARRAY (UTF-8 strings)");

  const int64_t nrows = meta->num_rows();
  if (nrows <= 0)
    throw std::runtime_error("empty parquet table: " + parquet_path.string());

  std::unordered_map<std::string, std::string> dict;
  {
    const std::size_t cap = static_cast<std::size_t>(nrows) * 2u;
    if (static_cast<int64_t>(cap / 2u) != nrows)
      throw std::runtime_error("sequence dict row count too large for memory reservation");
    dict.reserve(cap);
  }

  const int n_rg = meta->num_row_groups();
  std::size_t loaded = 0;
  for (int rg = 0; rg < n_rg; ++rg) {
    std::shared_ptr<parquet::RowGroupReader> rgr = pq_reader->RowGroup(rg);
    std::vector<std::string> keys;
    std::vector<std::string> vals;
    read_byte_array_column(rgr.get(), *pkey, 0, &keys);
    read_byte_array_column(rgr.get(), *pval, 1, &vals);
    if (keys.size() != vals.size())
      throw std::runtime_error("parquet row group " + std::to_string(rg) + ": key/value row count mismatch");
    for (std::size_t i = 0; i < keys.size(); ++i)
      dict.emplace(std::move(keys[i]), std::move(vals[i]));
    loaded += keys.size();
  }
  if (static_cast<int64_t>(loaded) != nrows)
    throw std::runtime_error("parquet rows read (" + std::to_string(loaded) + ") != footer num_rows (" +
                             std::to_string(nrows) + ")");

  return dict;
}

struct MinResult {
  std::string future_id;
  std::string best_current_id;
  int distance = -1;
};

static void usage(const char* prog) {
  std::cerr
      << "Usage: " << prog
      << " --parquet <sequence_dict.parquet> --data-dir <dir>\n"
         "\n"
         "<dir> must contain:\n"
         "  sample_ids_current.csv       — Sequence_ID or Sample_ID column (pandas; header optional)\n"
         "  sample_ids_future_<win>.csv  — e.g. sample_ids_future_3M_1.csv\n"
         "\n"
         "Output in <dir> (no header, comma-separated):\n"
         "  dist_<window>.csv  — Future_Sequence_ID,Current_Sequence_ID,Minimum_Distance\n";
}

int main(int argc, char** argv) {
  fs::path parquet_path;
  fs::path data_dir;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--parquet" && i + 1 < argc) {
      parquet_path = argv[++i];
    } else if (a == "--data-dir" && i + 1 < argc) {
      data_dir = argv[++i];
    } else if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    } else {
      std::cerr << "unknown argument: " << a << "\n";
      usage(argv[0]);
      return 1;
    }
  }
  if (parquet_path.empty() || data_dir.empty()) {
    usage(argv[0]);
    return 1;
  }

  try {
    if (!fs::is_directory(data_dir))
      throw std::runtime_error("data-dir is not a directory: " + data_dir.string());

    auto dict = load_sequence_dict_parquet(parquet_path);

    unsigned thread_cap = std::thread::hardware_concurrency();
    if (thread_cap == 0)
      thread_cap = 1;
    tbb::global_control tbb_threads(tbb::global_control::max_allowed_parallelism,
                                    static_cast<std::size_t>(thread_cap));

    const fs::path current_file = data_dir / "sample_ids_current.csv";
    auto current_ids = read_sample_ids_csv(current_file);
    if (current_ids.empty())
      throw std::runtime_error("no current sequence IDs in " + current_file.string());

    std::vector<std::string> current_seqs;
    current_seqs.reserve(current_ids.size());
    for (const auto& id : current_ids) {
      auto it = dict.find(id);
      if (it == dict.end())
        throw std::runtime_error("Sequence_ID not in parquet: " + id);
      current_seqs.push_back(it->second);
    }

    static constexpr const char kCurrentCsv[] = "sample_ids_current.csv";
    static constexpr const char kFuturePrefix[] = "sample_ids_future_";

    for (const auto& entry : fs::directory_iterator(data_dir)) {
      if (!entry.is_regular_file())
        continue;
      const fs::path p = entry.path();
      if (p.extension() != ".csv")
        continue;
      const std::string fname = p.filename().string();
      if (fname == kCurrentCsv)
        continue;
      if (fname.compare(0, sizeof(kFuturePrefix) - 1, kFuturePrefix) != 0)
        continue;

      const std::string stem = p.stem().string();
      const std::string window = stem.substr(sizeof(kFuturePrefix) - 1);
      if (window.empty()) {
        std::cerr << "skip future file with empty window tag: " << p << "\n";
        continue;
      }

      auto future_ids = read_sample_ids_csv(p);
      if (future_ids.empty()) {
        std::cerr << "skip empty future list: " << p << "\n";
        continue;
      }

      std::vector<std::string> future_seqs;
      future_seqs.reserve(future_ids.size());
      for (const auto& id : future_ids) {
        auto it = dict.find(id);
        if (it == dict.end())
          throw std::runtime_error("Sequence_ID not in parquet: " + id + " (" + p.string() + ")");
        future_seqs.push_back(it->second);
      }

      std::vector<MinResult> results(future_ids.size());
      for (std::size_t i = 0; i < future_ids.size(); ++i)
        results[i].future_id = future_ids[i];

      const std::size_t n_cur = current_seqs.size();
      const std::size_t n_fut = future_seqs.size();

      tbb::parallel_for(tbb::blocked_range<std::size_t>(0, n_fut),
                        [&](const tbb::blocked_range<std::size_t>& r) {
                          for (std::size_t fi = r.begin(); fi < r.end(); ++fi) {
                            const std::string& fseq = future_seqs[fi];
                            int best_d = std::numeric_limits<int>::max();
                            std::size_t best_ci = 0;
                            for (std::size_t ci = 0; ci < n_cur; ++ci) {
                              const int d = hamming_equal_length(fseq, current_seqs[ci]);
                              if (d < best_d) {
                                best_d = d;
                                best_ci = ci;
                                if (best_d == 0)
                                  break;
                              }
                            }
                            if (best_d == std::numeric_limits<int>::max())
                              throw std::runtime_error(
                                  "length mismatch: future sequence " + results[fi].future_id +
                                  " vs current pool (sequences must align to fixed length)");

                            results[fi].best_current_id = current_ids[best_ci];
                            results[fi].distance = best_d;
                          }
                        });

      const fs::path out_path = data_dir / ("dist_" + window + ".csv");
      std::ofstream out(out_path);
      if (!out)
        throw std::runtime_error("cannot write: " + out_path.string());

      for (const auto& r : results)
        out << r.future_id << ',' << r.best_current_id << ',' << r.distance << '\n';

      std::cout << "wrote " << out_path << " (" << results.size() << " rows)\n";
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "min_dist: " << e.what() << '\n';
    return 1;
  }
}
