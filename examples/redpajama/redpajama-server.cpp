#include <httplib.h>
#include <json.hpp>
#include "common-gptneox.h"
#include "gptneox.h"

struct server_params
{
  std::string hostname = "127.0.0.1";
  int32_t port = 8080;
};

struct gptneox_server_context
{
  bool as_loop = false;
  bool has_next_token = false;
  std::string generated_text = "";

  int32_t num_tokens_predicted = 0;
  int32_t n_past = 0;
  int32_t n_consumed = 0;
  int32_t n_session_consumed = 0;
  int32_t n_remain = 0;

  std::vector<gptneox_token> embd;
  std::vector<gptneox_token> last_n_tokens;
  std::vector<gptneox_token> processed_tokens;
  std::vector<gptneox_token> gptneox_token_newline;
  std::vector<gptneox_token> embd_inp;
  std::vector<std::vector<gptneox_token>> no_show_words;
  std::vector<gptneox_token> tokens_predicted;

  gptneox_context *ctx;
  gpt_params params;

  void rewind() {
    as_loop = false;
    params.antiprompt.clear();
    no_show_words.clear();
    num_tokens_predicted = 0;
    generated_text = "";
  }

  bool loadModel(gpt_params params_)
  {
    params = params_;
    auto lparams = gptneox_context_default_params();
    lparams.n_ctx      = params.n_ctx;
    lparams.n_parts    = params.n_parts;
    lparams.seed       = params.seed;
    lparams.f16_kv     = params.memory_f16;
    lparams.use_mmap   = params.use_mmap;
    lparams.use_mlock  = true;
    ctx = gptneox_init_from_file(params.model.c_str(), lparams);
    if (ctx == NULL)
    {
      fprintf(stderr, "%s: error: unable to load model\n", __func__);
      return false;
    }
    // determine newline token
    gptneox_token_newline = ::gptneox_tokenize(ctx, "\n", false);
    last_n_tokens.resize(params.n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    fprintf(stderr, "model was loaded");
    return true;
  }

  bool loadPrompt() {
    params.prompt.insert(0, 1, ' '); // always add a first space
    std::vector<gptneox_token> prompt_tokens = ::gptneox_tokenize(ctx, params.prompt, true);
    // compare the evaluated prompt with the new prompt
    int new_prompt_len = 0;
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
      if (i < processed_tokens.size() &&
        processed_tokens[i] == prompt_tokens[i])
      {
        continue;
      }
      else
      {
        embd_inp.push_back(prompt_tokens[i]);
        if(new_prompt_len == 0) {
          if(int32_t(i) - 1 < n_past) {
            processed_tokens.erase(processed_tokens.begin() + i, processed_tokens.end());
          }
          // Evaluate the new fragment prompt from the last token processed.
          n_past = processed_tokens.size();
        }
        new_prompt_len ++;
      }
    }
    if(n_past > 0 && params.interactive) {
      n_remain -= new_prompt_len;
    }
    if ((int)embd_inp.size() > params.n_ctx - 4)
    {
      return false;
    }
    has_next_token = true;
    fprintf(stderr, "%s", params.prompt.c_str());
    return true;
  }

  void beginCompletion()
  {
    if(n_remain == 0) {
      // number of tokens to keep when resetting context
      if (params.n_keep < 0 || params.n_keep > (int)embd_inp.size())
      {
        params.n_keep = (int)embd_inp.size();
      }
    }
    n_remain = params.n_predict;
  }

  gptneox_token nextToken() {
    gptneox_token result = -1;
    if (embd.size() > 0)
    {
      if (n_past + (int)embd.size() > params.n_ctx)
      {
        // Reset context
        const int n_left = n_past - params.n_keep;
        n_past = std::max(1, params.n_keep);
        processed_tokens.erase(processed_tokens.begin() + n_past, processed_tokens.end());
        embd.insert(embd.begin(), last_n_tokens.begin() + params.n_ctx - n_left / 2 - embd.size(), last_n_tokens.end() - embd.size());
      }
      for (int i = 0; i < (int)embd.size(); i += params.n_batch)
      {
        int n_eval = (int)embd.size() - i;
        if (n_eval > params.n_batch)
        {
          n_eval = params.n_batch;
        }
        if (gptneox_eval(ctx, &embd[i], n_eval, n_past, params.n_threads))
        {
          fprintf(stderr, "%s : failed to eval\n", __func__);
          has_next_token = false;
          return result;
        }
        n_past += n_eval;
      }
    }
    embd.clear();
    if ((int) embd_inp.size() <= n_consumed && has_next_token) {
        // out of user input, sample next token
        const float   temp            = params.temp;
        const int32_t top_k           = params.top_k <= 0 ? gptneox_n_vocab(ctx) : params.top_k;
        const float   top_p           = params.top_p;
        const float   tfs_z           = params.tfs_z;
        const float   typical_p       = params.typical_p;
        const int32_t repeat_last_n   = params.repeat_last_n < 0 ? params.n_ctx : params.repeat_last_n;
        const float   repeat_penalty  = params.repeat_penalty;
        const float   alpha_presence  = params.presence_penalty;
        const float   alpha_frequency = params.frequency_penalty;
        const int     mirostat        = params.mirostat;
        const float   mirostat_tau    = params.mirostat_tau;
        const float   mirostat_eta    = params.mirostat_eta;
        const bool    penalize_nl     = params.penalize_nl;

        gptneox_token id = 0;

        {
            auto logits = gptneox_get_logits(ctx);
            auto n_vocab = gptneox_n_vocab(ctx);

            // Apply params.logit_bias map
            for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
                logits[it->first] += it->second;
            }

            std::vector<gptneox_token_data> candidates;
            candidates.reserve(n_vocab);
            for (gptneox_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(gptneox_token_data{token_id, logits[token_id], 0.0f});
            }

            gptneox_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            // Apply penalties
            gptneox_token nl_token = gptneox_str_to_token(ctx, "\n");
            float nl_logit = logits[nl_token];
            auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), params.n_ctx);
            gptneox_sample_repetition_penalty(ctx, &candidates_p,
                last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                last_n_repeat, repeat_penalty);
            gptneox_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                last_n_repeat, alpha_frequency, alpha_presence);
            if (!penalize_nl) {
                logits[nl_token] = nl_logit;
            }

            if (temp <= 0) {
                // Greedy sampling
                id = gptneox_sample_token_greedy(ctx, &candidates_p);
            } else {
                if (mirostat == 1) {
                    static float mirostat_mu = 2.0f * mirostat_tau;
                    const int mirostat_m = 100;
                    gptneox_sample_temperature(ctx, &candidates_p, temp);
                    id = gptneox_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                } else if (mirostat == 2) {
                    static float mirostat_mu = 2.0f * mirostat_tau;
                    gptneox_sample_temperature(ctx, &candidates_p, temp);
                    id = gptneox_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                } else {
                    // Temperature sampling
                    gptneox_sample_top_k(ctx, &candidates_p, top_k, 1);
                    gptneox_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
                    gptneox_sample_typical(ctx, &candidates_p, typical_p, 1);
                    gptneox_sample_top_p(ctx, &candidates_p, top_p, 1);
                    gptneox_sample_temperature(ctx, &candidates_p, temp);
                    id = gptneox_sample_token(ctx, &candidates_p);
                }
            }
            // printf("`%d`", candidates_p.size);

            last_n_tokens.erase(last_n_tokens.begin());
            last_n_tokens.push_back(id);
        }

        // replace end of text token with newline token when in interactive mode
        if (id == gptneox_token_eos() && params.interactive && !params.instruct) {
            id = gptneox_token_newline.front();
            if (params.antiprompt.size() != 0) {
                // tokenize and inject first reverse prompt
                const auto first_antiprompt = ::gptneox_tokenize(ctx, params.antiprompt.front(), false);
                embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
            }
        }

        // add it to the context
        embd.push_back(id);
        for (auto id : embd)
        {
          result = id;
        }
        // decrement remaining sampling budget
        --n_remain;
    } else {
        // some user input remains from prompt or interaction, forward it to processing
        while ((int) embd_inp.size() > n_consumed) {
            embd.push_back(embd_inp[n_consumed]);
            last_n_tokens.erase(last_n_tokens.begin());
            last_n_tokens.push_back(embd_inp[n_consumed]);
            ++n_consumed;
            if ((int) embd.size() >= params.n_batch) {
                break;
            }
        }
    }
    if (params.interactive && (int)embd_inp.size() <= n_consumed)
    {
      // check for reverse prompt
      if (params.antiprompt.size())
      {
        std::string last_output;
        for (auto id : last_n_tokens)
        {
          last_output += gptneox_token_to_str(ctx, id);
        }
        has_next_token = true;
        // Check if each of the reverse prompts appears at the end of the output.
        for (std::string &antiprompt : params.antiprompt)
        {
          if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos)
          {
            has_next_token = false;
            return result;
          }
        }
      }
      if (n_past > 0)
      {
        has_next_token = true;
      }
    }

    if (!embd.empty() && embd.back() == gptneox_token_eos()) {
        has_next_token = false;
    }

    if (params.interactive && n_remain <= 0 && params.n_predict != -1)
    {
      n_remain = params.n_predict;
    }
    has_next_token = n_remain != 0;
    return result;
  }

  std::string doCompletion()
  {
    gptneox_token token = nextToken();

    if (token == -1) {
      return "";
    }
    tokens_predicted.clear();
    tokens_predicted.push_back(token);

    // Avoid add the no show words to the response
    for (std::vector<gptneox_token> word_tokens : no_show_words)
    {
      size_t match_token = 1;
      if (tokens_predicted.front() == word_tokens.front())
      {
        bool execute_matching = true;
        if (tokens_predicted.size() > 1) { // if previus tokens had been tested
          for (size_t i = 1; i < word_tokens.size(); i++)
          {
            if (i >= tokens_predicted.size()) {
              match_token = i;
              break;
            }
            if (tokens_predicted[i] == word_tokens[i])
            {
              continue;
            }
            else
            {
              execute_matching = false;
              break;
            }
          }
        }
        while (execute_matching) {
          if (match_token == word_tokens.size()) {
            return "";
          }
          token = nextToken();
          tokens_predicted.push_back(token);
          if (token == word_tokens[match_token])
          { // the token follow the sequence
            match_token++;
          }
          else if (match_token < word_tokens.size())
          { // no complete all word sequence
            break;
          }
        }
      }
    }
    if(as_loop) {
      generated_text = "";
    }
    for (gptneox_token tkn : tokens_predicted)
    {
      generated_text += gptneox_token_to_str(ctx, tkn);
    }
    return generated_text;
  }

  std::vector<float> embedding(std::string content, int threads) {
    content.insert(0, 1, ' ');
    std::vector<gptneox_token> tokens = ::gptneox_tokenize(ctx, content, true);
    if (tokens.size() > 0)
    {
      if (gptneox_eval(ctx, tokens.data(), tokens.size(), 0, threads))
      {
        fprintf(stderr, "%s : failed to eval\n", __func__);
        std::vector<float> embeddings_;
        return embeddings_;
      }
    }
    const int n_embd = gptneox_n_embd(ctx);
    const auto embeddings = gptneox_get_embeddings(ctx);
    std::vector<float> embeddings_(embeddings, embeddings + n_embd);
    return embeddings_;
  }
};

using namespace httplib;

using json = nlohmann::json;

void server_print_usage(int /*argc*/, char **argv, const gpt_params &params)
{
  fprintf(stderr, "usage: %s [options]\n", argv[0]);
  fprintf(stderr, "\n");
  fprintf(stderr, "options:\n");
  fprintf(stderr, "  -h, --help            show this help message and exit\n");
  fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1, use random seed for < 0)\n");
  fprintf(stderr, "  -c N, --ctx-size N    size of the prompt context (default: %d)\n", params.n_ctx);
  fprintf(stderr, "  --memory-f32          use f32 instead of f16 for memory key+value (default: disabled)\n");
  fprintf(stderr, "                        not recommended: doubles context memory required and no measurable increase in quality\n");
  fprintf(stderr, "  --embedding           enable embedding mode\n");
  fprintf(stderr, "  --keep                number of tokens to keep from the initial prompt (default: %d, -1 = all)\n", params.n_keep);
  if (gptneox_mlock_supported())
  {
    fprintf(stderr, "  --mlock               force system to keep model in RAM rather than swapping or compressing\n");
  }
  if (gptneox_mmap_supported())
  {
    fprintf(stderr, "  --no-mmap             do not memory-map model (slower load but may reduce pageouts if not using mlock)\n");
  }
  fprintf(stderr, "  -m FNAME, --model FNAME\n");
  fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
  fprintf(stderr, "  -a ALIAS, --alias ALIAS\n");
  fprintf(stderr, "                        set an alias for the model, will be added as `model` field in completion response\n");
  fprintf(stderr, "  --host                ip address to listen (default 127.0.0.1)\n");
  fprintf(stderr, "  --port PORT           port to listen (default 8080)\n");
  fprintf(stderr, "\n");
}

bool server_params_parse(int argc, char **argv, server_params &sparams, gpt_params &params)
{
  gpt_params default_params;
  std::string arg;
  bool invalid_param = false;

  for (int i = 1; i < argc; i++)
  {
    arg = argv[i];
    if (arg == "--port")
    {
      if (++i >= argc)
      {
        invalid_param = true;
        break;
      }
      sparams.port = std::stoi(argv[i]);
    }
    else if (arg == "--host")
    {
      if (++i >= argc)
      {
        invalid_param = true;
        break;
      }
      sparams.hostname = argv[i];
    }
    else if (arg == "-s" || arg == "--seed")
    {
#if defined(GGML_USE_CUBLAS)
      fprintf(stderr, "WARNING: when using cuBLAS generation results are NOT guaranteed to be reproducible.\n");
#endif
      if (++i >= argc)
      {
        invalid_param = true;
        break;
      }
      params.seed = std::stoi(argv[i]);
    }
    else if (arg == "-m" || arg == "--model")
    {
      if (++i >= argc)
      {
        invalid_param = true;
        break;
      }
      params.model = argv[i];
    }
    else if (arg == "--embedding")
    {
      params.embedding = true;
    }
    else if (arg == "-h" || arg == "--help")
    {
      server_print_usage(argc, argv, default_params);
      exit(0);
    }
    else if (arg == "-c" || arg == "--ctx-size" || arg == "--ctx_size")
    {
      if (++i >= argc)
      {
        invalid_param = true;
        break;
      }
      params.n_ctx = std::stoi(argv[i]);
    }
    else if (arg == "--memory-f32" || arg == "--memory_f32")
    {
      params.memory_f16 = false;
    }
    else if (arg == "--gpu-layers" || arg == "-ngl" || arg == "--n-gpu-layers")
    {
      if (++i >= argc)
      {
        invalid_param = true;
        break;
      }
    }
    else
    {
      fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
      server_print_usage(argc, argv, default_params);
      exit(1);
    }
  }

  if (invalid_param)
  {
    fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
    server_print_usage(argc, argv, default_params);
    exit(1);
  }
  return true;
}

bool parse_options_completion(json body, gptneox_server_context& gptneox, Response &res) {
  if (!body["threads"].is_null())
  {
    gptneox.params.n_threads = body["threads"].get<int>();
  }
  if (!body["n_predict"].is_null())
  {
    gptneox.params.n_predict = body["n_predict"].get<int>();
  }
  if (!body["top_k"].is_null())
  {
    gptneox.params.top_k = body["top_k"].get<int>();
  }
  if (!body["top_p"].is_null())
  {
    gptneox.params.top_p = body["top_p"].get<float>();
  }
  if (!body["temperature"].is_null())
  {
    gptneox.params.temp = body["temperature"].get<float>();
  }
  if (!body["batch_size"].is_null())
  {
    gptneox.params.n_batch = body["batch_size"].get<int>();
  }
  if (!body["n_keep"].is_null())
  {
    gptneox.params.n_keep = body["n_keep"].get<int>();
  }
  if (!body["as_loop"].is_null())
  {
    gptneox.as_loop = body["as_loop"].get<bool>();
  }
  if (!body["interactive"].is_null())
  {
    gptneox.params.interactive = body["interactive"].get<bool>();
  }
  if (!body["prompt"].is_null())
  {
    gptneox.params.prompt = body["prompt"].get<std::string>();
  }
  else
  {
    json data = {
        {"status", "error"},
        {"reason", "You need to pass the prompt"}};
    res.set_content(data.dump(), "application/json");
    res.status = 400;
    return false;
  }
  if (!body["stop"].is_null())
  {
    std::vector<std::string> stop_words = body["stop"].get<std::vector<std::string>>();
    for (std::string stop_word : stop_words)
    {
      gptneox.params.antiprompt.push_back(stop_word);
      gptneox.no_show_words.push_back(::gptneox_tokenize(gptneox.ctx, stop_word, false));
    }
  }
  if (!body["exclude"].is_null())
  {
    std::vector<std::string> no_show_words = body["exclude"].get<std::vector<std::string>>();
    for (std::string no_show : no_show_words)
    {
      gptneox.no_show_words.push_back(::gptneox_tokenize(gptneox.ctx, no_show, false));
    }
  }
  return true;
}

int main(int argc, char **argv)
{
  // own arguments required by this example
  gpt_params params;
  server_params sparams;

  // struct that contains gptneox context and inference
  gptneox_server_context redpajama;
  params.model = "ggml-model.bin";

  if (server_params_parse(argc, argv, sparams, params) == false)
  {
    return 1;
  }

  if (params.seed <= 0)
  {
    params.seed = time(NULL);
  }

  fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

  // load the model
  if (!redpajama.loadModel(params))
  {
    return 1;
  }

  Server svr;

  svr.Get("/", [](const Request &, Response &res){
            res.set_content("<h1>redpajama.cpp server works</h1>", "text/html");
          });

  svr.Post("/completion", [&redpajama](const Request &req, Response &res){
              if(redpajama.params.embedding) {
                json data = {
                    {"status", "error"},
                    {"reason", "To use completion function disable embedding mode"}};
                res.set_content(data.dump(), "application/json");
                res.status = 400;
                fprintf(stderr, "To use completion function disable embedding mode");
                return;
              }

              redpajama.rewind();

              if(parse_options_completion(json::parse(req.body), redpajama, res) == false){
                return;
              }

              if (!redpajama.loadPrompt())
              {
                json data = {
                    {"status", "error"},
                    {"reason", "Context too long, please be more specific"}};
                res.set_content(data.dump(), "application/json");
                res.status = 400;
                fprintf(stderr, "Context too long, please be more specific");
                return;
              }

              redpajama.beginCompletion();
              if(redpajama.as_loop) {
                json data = {
                    {"status", "done" } };
                return res.set_content(data.dump(), "application/json");
              } else {
                // loop inference until finish completion
                while (redpajama.has_next_token)
                {
                  redpajama.doCompletion();
                }
                try
                {
                  json data = {
                      {"content", redpajama.generated_text },
                      {"tokens_predicted", redpajama.num_tokens_predicted}};
                  return res.set_content(data.dump(), "application/json");
                }
                catch (const json::exception &e)
                {
                  // Some tokens have bad UTF-8 strings, the json parser is very sensitive
                  json data = {
                      {"content", "Bad encoding token"},
                      {"tokens_predicted", 0}};
                  fprintf(stderr, "Bad encoding token");
                  return res.set_content(data.dump(), "application/json");
                }
              } });

  svr.Post("/tokenize", [&redpajama](const Request &req, Response &res)
            {
              json body = json::parse(req.body);
              json data = {
                    {"tokens", ::gptneox_tokenize(redpajama.ctx, body["content"].get<std::string>(), false) } };
                return res.set_content(data.dump(), "application/json");
            });

  svr.Post("/embedding", [&redpajama](const Request &req, Response &res)
            {
              if(!redpajama.params.embedding) {
                std::vector<float> empty;
                json data = {
                    {"embedding", empty}};
                fprintf(stderr, "[redpajama-server] : You need enable embedding mode adding: --embedding option\n");
                return res.set_content(data.dump(), "application/json");
              }
              json body = json::parse(req.body);
              std::string content = body["content"].get<std::string>();
              int threads = body["threads"].get<int>();
              json data = {
                    {"embedding", redpajama.embedding(content, threads) } };
              return res.set_content(data.dump(), "application/json");
            });

  svr.Get("/next-token", [&redpajama](const Request &req, Response &res)
          {
            if(redpajama.params.embedding) {
                res.set_content("{}", "application/json");
                return;
            }
            std::string result = "";
            if (req.has_param("stop")) {
                redpajama.has_next_token = false;
            } else {
              result = redpajama.doCompletion(); // inference next token
            }
            try {
              json data = {
                        {"content", result },
                        {"stop", !redpajama.has_next_token }};
              return res.set_content(data.dump(), "application/json");
            } catch (const json::exception &e) {
              // Some tokens have bad UTF-8 strings, the json parser is very sensitive
              json data = {
                        {"content", "" },
                        {"stop", !redpajama.has_next_token }};
              return res.set_content(data.dump(), "application/json");
            }
          });

  fprintf(stderr, "%s: http server Listening at http://%s:%i\n", __func__, sparams.hostname.c_str(), sparams.port);

  if(params.embedding) {
    fprintf(stderr, "NOTE: Mode embedding enabled. Completion function doesn't work in this mode.\n");
  }

  // change hostname and port
  svr.listen(sparams.hostname, sparams.port);
}
