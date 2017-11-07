/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "fasttext.h"

#include <math.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <stdexcept>


namespace fasttext {

FastText::FastText() : quant_(false) {}

void FastText::addInputVector(Vector& vec, int32_t ind) const {
  if (quant_) {
    vec.addRow(*qinput_, ind);
  } else {
    vec.addRow(*input_, ind);
  }
}

std::shared_ptr<const Dictionary> FastText::getDictionary() const {
  return dict_;
}

const Args FastText::getArgs() const {
  return *args_.get();
}

std::shared_ptr<const Matrix> FastText::getInputMatrix() const {
  return input_;
}

std::shared_ptr<const Matrix> FastText::getOutputMatrix() const {
  return output_;
}

int32_t FastText::getWordId(const std::string& word) const {
  return dict_->getId(word);
}

int32_t FastText::getSubwordId(const std::string& word) const {
  int32_t h = dict_->hash(word) % args_->bucket;
  return dict_->nwords() + h;
}

void FastText::getWordVector(Vector& vec, const std::string& word) const {
  const std::vector<int32_t>& ngrams = dict_->getSubwords(word);
  vec.zero();
  for (int i = 0; i < ngrams.size(); i ++) {
    addInputVector(vec, ngrams[i]);
  }
  if (ngrams.size() > 0) {
    vec.mul(1.0 / ngrams.size());
  }
}

void FastText::getVector(Vector& vec, const std::string& word) const {
  getWordVector(vec, word);
}

void FastText::getSubwordVector(Vector& vec, const std::string& subword)
    const {
  vec.zero();
  int32_t h = dict_->hash(subword) % args_->bucket;
  h = h + dict_->nwords();
  addInputVector(vec, h);
}

void FastText::saveVectors() {
  std::ofstream ofs(args_->output + ".vec");
  if (!ofs.is_open()) {
    std::cerr << "Error opening file for saving vectors." << std::endl;
    exit(EXIT_FAILURE);
  }
  ofs << dict_->nwords() << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    getWordVector(vec, word);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

void FastText::saveOutput() {
  std::ofstream ofs(args_->output + ".output");
  if (!ofs.is_open()) {
    std::cerr << "Error opening file for saving vectors." << std::endl;
    exit(EXIT_FAILURE);
  }
  if (quant_) {
    std::cerr << "Option -saveOutput is not supported for quantized models."
              << std::endl;
    return;
  }
  int32_t n = (args_->model == model_name::sup) ? dict_->nlabels()
                                                : dict_->nwords();
  ofs << n << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < n; i++) {
    std::string word = (args_->model == model_name::sup) ? dict_->getLabel(i)
                                                         : dict_->getWord(i);
    vec.zero();
    vec.addRow(*output_, i);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

bool FastText::checkModel(std::istream& in) {
  int32_t magic;
  in.read((char*)&(magic), sizeof(int32_t));
  if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT32) {
    return false;
  }
  in.read((char*)&(version), sizeof(int32_t));
  if (version > FASTTEXT_VERSION) {
    return false;
  }
  return true;
}

void FastText::signModel(std::ostream& out) {
  const int32_t magic = FASTTEXT_FILEFORMAT_MAGIC_INT32;
  const int32_t version = FASTTEXT_VERSION;
  out.write((char*)&(magic), sizeof(int32_t));
  out.write((char*)&(version), sizeof(int32_t));
}

void FastText::saveModel() {
  std::string fn(args_->output);
  if (quant_) {
    fn += ".ftz";
  } else {
    fn += ".bin";
  }
  saveModel(fn);
}

void FastText::saveModel(const std::string path) {
  std::ofstream ofs(path, std::ofstream::binary);
  if (!ofs.is_open()) {
    throw std::invalid_argument(path + " cannot be opened for saving!");
  }
  signModel(ofs);
  args_->save(ofs);
  dict_->save(ofs);

  ofs.write((char*)&(quant_), sizeof(bool));
  if (quant_) {
    qinput_->save(ofs);
  } else {
    input_->save(ofs);
  }

  ofs.write((char*)&(args_->qout), sizeof(bool));
  if (quant_ && args_->qout) {
    qoutput_->save(ofs);
  } else {
    output_->save(ofs);
  }

  ofs.close();
}

void FastText::loadModel(const std::string& filename) {
  std::ifstream ifs(filename, std::ifstream::binary);
  if (!ifs.is_open()) {
    throw std::invalid_argument(filename + " cannot be opened for loading!");
  }
  if (!checkModel(ifs)) {
    throw std::invalid_argument(filename + " has wrong file format!");
  }
  loadModel(ifs);
  ifs.close();
}

void FastText::loadModel(std::istream& in) {
  args_ = std::make_shared<Args>();
  dict_ = std::make_shared<Dictionary>(args_);
  input_ = std::make_shared<Matrix>();
  output_ = std::make_shared<Matrix>();
  qinput_ = std::make_shared<QMatrix>();
  qoutput_ = std::make_shared<QMatrix>();
  args_->load(in);
  if (version == 11 && args_->model == model_name::sup) {
    // backward compatibility: old supervised models do not use char ngrams.
    args_->maxn = 0;
  }
  dict_->load(in);

  bool quant_input;
  in.read((char*) &quant_input, sizeof(bool));
  if (quant_input) {
    quant_ = true;
    qinput_->load(in);
  } else {
    input_->load(in);
  }

  if (!quant_input && dict_->isPruned()) {
    std::cerr << "Invalid model file.\n"
              << "Please download the updated model from www.fasttext.cc.\n"
              << "See issue #332 on Github for more information.\n";
    exit(1);
  }

  in.read((char*) &args_->qout, sizeof(bool));
  if (quant_ && args_->qout) {
    qoutput_->load(in);
  } else {
    output_->load(in);
  }

  model_ = std::make_shared<Model>(input_, output_, args_, 0);
  model_->quant_ = quant_;
  model_->setQuantizePointer(qinput_, qoutput_, args_->qout);

  if (args_->model == model_name::sup) {
    model_->setTargetCounts(dict_->getCounts(entry_type::label));
  } else {
    model_->setTargetCounts(dict_->getCounts(entry_type::word));
  }
}

void FastText::printInfo(real progress, real loss) {
  real t = real(clock() - start) / CLOCKS_PER_SEC;
  real wst = real(tokenCount) / t;
  real lr = args_->lr * (1.0 - progress);
  int eta = int(t / progress * (1 - progress) / args_->thread);
  int etah = eta / 3600;
  int etam = (eta - etah * 3600) / 60;
  std::cerr << std::fixed;
  std::cerr << "\rProgress: " << std::setprecision(1) << 100 * progress << "%";
  std::cerr << "  words/sec/thread: " << std::setprecision(0) << wst;
  std::cerr << "  lr: " << std::setprecision(6) << lr;
  std::cerr << "  loss: " << std::setprecision(6) << loss;
  std::cerr << "  eta: " << etah << "h" << etam << "m ";
  std::cerr << std::flush;
}

std::vector<int32_t> FastText::selectEmbeddings(int32_t cutoff) const {
  Vector norms(input_->m_);
  input_->l2NormRow(norms);
  std::vector<int32_t> idx(input_->m_, 0);
  std::iota(idx.begin(), idx.end(), 0);
  auto eosid = dict_->getId(Dictionary::EOS);
  std::sort(idx.begin(), idx.end(),
      [&norms, eosid] (size_t i1, size_t i2) {
      return eosid ==i1 || (eosid != i2 && norms[i1] > norms[i2]);
      });
  idx.erase(idx.begin() + cutoff, idx.end());
  return idx;
}

void FastText::quantize(std::shared_ptr<Args> qargs) {
  if (args_->model != model_name::sup) {
    throw std::invalid_argument(
        "For now we only support quantization of supervised models");
  }
  args_->input = qargs->input;
  args_->qout = qargs->qout;
  args_->output = qargs->output;

  if (qargs->cutoff > 0 && qargs->cutoff < input_->m_) {
    auto idx = selectEmbeddings(qargs->cutoff);
    dict_->prune(idx);
    std::shared_ptr<Matrix> ninput =
        std::make_shared<Matrix>(idx.size(), args_->dim);
    for (auto i = 0; i < idx.size(); i++) {
      for (auto j = 0; j < args_->dim; j++) {
        ninput->at(i, j) = input_->at(idx[i], j);
      }
    }
    input_ = ninput;
    if (qargs->retrain) {
      args_->epoch = qargs->epoch;
      args_->lr = qargs->lr;
      args_->thread = qargs->thread;
      args_->verbose = qargs->verbose;
      startThreads();
    }
  }

  qinput_ = std::make_shared<QMatrix>(*input_, qargs->dsub, qargs->qnorm);

  if (args_->qout) {
    qoutput_ = std::make_shared<QMatrix>(*output_, 2, qargs->qnorm);
  }

  quant_ = true;
  model_ = std::make_shared<Model>(input_, output_, args_, 0);
  model_->quant_ = quant_;
  model_->setQuantizePointer(qinput_, qoutput_, args_->qout);
}

void FastText::supervised(
    Model& model,
    real lr,
    const std::vector<int32_t>& line,
    const std::vector<int32_t>& labels) {
  if (labels.size() == 0 || line.size() == 0) return;
  /*
   *因为一个句子可以打上多个 label，但是 fastText 的架构实际上只有支持一个 label
    所以这里随机选择一个 label 来更新模型，这样做会让其它 label 被忽略
    所以 fastText 不太适合做多标签的分类
    (kexin:原来是这样来实现多分类的)
   */
  std::uniform_int_distribution<> uniform(0, labels.size() - 1);
  int32_t i = uniform(model.rng);
  model.update(line, labels[i], lr);
}

// cbow: 用周围的词预测当前词
void FastText::cbow(Model& model, real lr,
                    const std::vector<int32_t>& line) {
  std::vector<int32_t> bow;
  std::uniform_int_distribution<> uniform(1, args_->ws);
  // 在一个句子中，每个词可以进行一次 update
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(model.rng);
    bow.clear();
    // 以当前词为中心，将左右 boundary 个词加入 input
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        // 实际被加入 input 的不止是词本身，还有词的 word n-gram
        const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w + c]);
        bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
      }
    }
    // 完成一次 CBOW 更新
    model.update(bow, line[w], lr);
  }
}

// 用当前词预测周围的词
void FastText::skipgram(Model& model, real lr,
                        const std::vector<int32_t>& line) {
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    // 一个词的上下文长度是随机产生的
    int32_t boundary = uniform(model.rng);
    // 采用词+word n-gram 来预测这个词的上下文的所有的词
    const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w]);
    // 在 skipgram 中，对上下文的每一个词分别更新一次模型
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        model.update(ngrams, line[w + c], lr);
      }
    }
  }
}

void FastText::test(std::istream& in, int32_t k) {
  int32_t nexamples = 0, nlabels = 0;
  double precision = 0.0;
  std::vector<int32_t> line, labels;

  while (in.peek() != EOF) {
    dict_->getLine(in, line, labels, model_->rng);
    if (labels.size() > 0 && line.size() > 0) {
      std::vector<std::pair<real, int32_t>> modelPredictions;
      model_->predict(line, k, modelPredictions);
      for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++) {
        if (std::find(labels.begin(), labels.end(), it->second) != labels.end()) {
          precision += 1.0;
        }
      }
      nexamples++;
      nlabels += labels.size();
    }
  }
  std::cout << "N" << "\t" << nexamples << std::endl;
  std::cout << std::setprecision(3);
  std::cout << "P@" << k << "\t" << precision / (k * nexamples) << std::endl;
  std::cout << "R@" << k << "\t" << precision / nlabels << std::endl;
  std::cerr << "Number of examples: " << nexamples << std::endl;
}

void FastText::predict(std::istream& in, int32_t k,
                       std::vector<std::pair<real,std::string>>& predictions) const {
  std::vector<int32_t> words, labels;
  predictions.clear();
  dict_->getLine(in, words, labels, model_->rng);
  predictions.clear();
  if (words.empty()) return;
  Vector hidden(args_->dim);
  Vector output(dict_->nlabels());
  std::vector<std::pair<real,int32_t>> modelPredictions;
  // 调用 model 模块的预测接口，获取 k 个最可能的分类
  model_->predict(words, k, modelPredictions, hidden, output);
  for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend(); it++) {
    predictions.push_back(std::make_pair(it->first, dict_->getLabel(it->second)));
  }
}

void FastText::predict(std::istream& in, int32_t k, bool print_prob) {
  std::vector<std::pair<real,std::string>> predictions;
  while (in.peek() != EOF) {
    predictions.clear();
    predict(in, k, predictions);
    if (predictions.empty()) {
      std::cout << std::endl;
      continue;
    }
    for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
      if (it != predictions.cbegin()) {
        std::cout << " ";
      }
      std::cout << it->second;
      if (print_prob) {
        std::cout << " " << exp(it->first);
      }
    }
    std::cout << std::endl;
  }
}

void FastText::getSentenceVector(
    std::istream& in,
    fasttext::Vector& svec) {
  svec.zero();
  if (args_->model == model_name::sup) {
    std::vector<int32_t> line, labels;
    dict_->getLine(in, line, labels, model_->rng);
    for (int32_t i = 0; i < line.size(); i++) {
      addInputVector(svec, line[i]);
    }
    if (!line.empty()) {
      svec.mul(1.0 / line.size());
    }
  } else {
    Vector vec(args_->dim);
    std::string sentence;
    std::getline(in, sentence);
    std::istringstream iss(sentence);
    std::string word;
    int32_t count = 0;
    while (iss >> word) {
      getWordVector(vec, word);
      real norm = vec.norm();
      if (norm > 0) {
        vec.mul(1.0 / norm);
        svec.addVector(vec);
        count++;
      }
    }
    if (count > 0) {
      svec.mul(1.0 / count);
    }
  }
}

void FastText::ngramVectors(std::string word) {
  std::vector<int32_t> ngrams;
  std::vector<std::string> substrings;
  Vector vec(args_->dim);
  dict_->getSubwords(word, ngrams, substrings);
  for (int32_t i = 0; i < ngrams.size(); i++) {
    vec.zero();
    if (ngrams[i] >= 0) {
      if (quant_) {
        vec.addRow(*qinput_, ngrams[i]);
      } else {
        vec.addRow(*input_, ngrams[i]);
      }
    }
    std::cout << substrings[i] << " " << vec << std::endl;
  }
}

void FastText::precomputeWordVectors(Matrix& wordVectors) {
  Vector vec(args_->dim);
  wordVectors.zero();
  std::cerr << "Pre-computing word vectors...";
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    getWordVector(vec, word);
    real norm = vec.norm();
    if (norm > 0) {
      wordVectors.addRow(vec, i, 1.0 / norm);
    }
  }
  std::cerr << " done." << std::endl;
}

void FastText::findNN(const Matrix& wordVectors, const Vector& queryVec,
                      int32_t k, const std::set<std::string>& banSet) {
  real queryNorm = queryVec.norm();
  if (std::abs(queryNorm) < 1e-8) {
    queryNorm = 1;
  }
  std::priority_queue<std::pair<real, std::string>> heap;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    real dp = wordVectors.dotRow(queryVec, i);
    heap.push(std::make_pair(dp / queryNorm, word));
  }
  int32_t i = 0;
  while (i < k && heap.size() > 0) {
    auto it = banSet.find(heap.top().second);
    if (it == banSet.end()) {
      std::cout << heap.top().second << " " << heap.top().first << std::endl;
      i++;
    }
    heap.pop();
  }
}

void FastText::nn(int32_t k) {
  std::string queryWord;
  Vector queryVec(args_->dim);
  Matrix wordVectors(dict_->nwords(), args_->dim);
  precomputeWordVectors(wordVectors);
  std::set<std::string> banSet;
  std::cout << "Query word? ";
  while (std::cin >> queryWord) {
    banSet.clear();
    banSet.insert(queryWord);
    getWordVector(queryVec, queryWord);
    findNN(wordVectors, queryVec, k, banSet);
    std::cout << "Query word? ";
  }
}

void FastText::analogies(int32_t k) {
  std::string word;
  Vector buffer(args_->dim), query(args_->dim);
  Matrix wordVectors(dict_->nwords(), args_->dim);
  precomputeWordVectors(wordVectors);
  std::set<std::string> banSet;
  std::cout << "Query triplet (A - B + C)? ";
  while (true) {
    banSet.clear();
    query.zero();
    std::cin >> word;
    banSet.insert(word);
    getWordVector(buffer, word);
    query.addVector(buffer, 1.0);
    std::cin >> word;
    banSet.insert(word);
    getWordVector(buffer, word);
    query.addVector(buffer, -1.0);
    std::cin >> word;
    banSet.insert(word);
    getWordVector(buffer, word);
    query.addVector(buffer, 1.0);

    findNN(wordVectors, query, k, banSet);
    std::cout << "Query triplet (A - B + C)? ";
  }
}

void FastText::trainThread(int32_t threadId) {
  std::ifstream ifs(args_->input);
  // 根据线程数，将训练文件按照总字节数（utils::size）均分成多个部分
  // 这么做的一个后果是，每一部分的第一个词有可能从中间被切断，
  // 这样的"小噪音"对于整体的训练结果无影响
  utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);

  Model model(input_, output_, args_, threadId);
  if (args_->model == model_name::sup) {
    model.setTargetCounts(dict_->getCounts(entry_type::label));
  } else {
    model.setTargetCounts(dict_->getCounts(entry_type::word));
  }

  // 训练文件中的 token 总数
  const int64_t ntokens = dict_->ntokens();
  // 当前线程处理完毕的 token 总数
  int64_t localTokenCount = 0;
  std::vector<int32_t> line, labels;
  // tokenCount 为所有线程处理完毕的 token 总数
  // 当处理了 args_->epoch 遍所有 token 后，训练结束
  while (tokenCount < args_->epoch * ntokens) {
    // progress = 0 ~ 1，代表当前训练进程，随着训练的进行逐渐增大
    real progress = real(tokenCount) / (args_->epoch * ntokens);
    // 学习率根据 progress 线性下降
    real lr = args_->lr * (1.0 - progress);
    // 根据训练需求的不同，这里用的更新策略也不同，它们分别是：
    // 1. 有监督学习（分类）, sup即为supervised
    if (args_->model == model_name::sup) {
      localTokenCount += dict_->getLine(ifs, line, labels, model.rng);
      supervised(model, lr, line, labels);
    } else if (args_->model == model_name::cbow) {
      localTokenCount += dict_->getLine(ifs, line, model.rng);
      cbow(model, lr, line);
    } else if (args_->model == model_name::sg) {
      localTokenCount += dict_->getLine(ifs, line, model.rng);
      skipgram(model, lr, line);
    }

    // args_->lrUpdateRate 是每个线程学习率的变化率，默认为 100，
    // 它的作用是，每处理一定的行数，再更新全局的 tokenCount 变量，从而影响学习率
    if (localTokenCount > args_->lrUpdateRate) {
      tokenCount += localTokenCount;
      // 每次更新 tokenCount 后，重置计数
      localTokenCount = 0;
      // 0 号线程负责将训练进度输出到屏幕
      if (threadId == 0 && args_->verbose > 1) {
        printInfo(progress, model.getLoss());
      }
    }
  }

  if (threadId == 0 && args_->verbose > 0) {
    printInfo(1.0, model.getLoss());
    std::cerr << std::endl;
  }
  ifs.close();
}

void FastText::loadVectors(std::string filename) {
  std::ifstream in(filename);
  std::vector<std::string> words;
  std::shared_ptr<Matrix> mat; // temp. matrix for pretrained vectors
  int64_t n, dim;
  if (!in.is_open()) {
    std::cerr << "Pretrained vectors file cannot be opened!" << std::endl;
    exit(EXIT_FAILURE);
  }
  in >> n >> dim;
  if (dim != args_->dim) {
    std::cerr << "Dimension of pretrained vectors does not match -dim option"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  mat = std::make_shared<Matrix>(n, dim);
  for (size_t i = 0; i < n; i++) {
    std::string word;
    in >> word;
    words.push_back(word);
    dict_->add(word);
    for (size_t j = 0; j < dim; j++) {
      in >> mat->data_[i * dim + j];
    }
  }
  in.close();

  dict_->threshold(1, 0);
  input_ = std::make_shared<Matrix>(dict_->nwords()+args_->bucket, args_->dim);
  input_->uniform(1.0 / args_->dim);

  for (size_t i = 0; i < n; i++) {
    int32_t idx = dict_->getId(words[i]);
    if (idx < 0 || idx >= dict_->nwords()) continue;
    for (size_t j = 0; j < dim; j++) {
      input_->data_[idx * dim + j] = mat->data_[i * dim + j];
    }
  }
}

void FastText::train(std::shared_ptr<Args> args) {
  args_ = args;
  dict_ = std::make_shared<Dictionary>(args_);
  if (args_->input == "-") {
    // manage expectations
    std::cerr << "Cannot use stdin for training!" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::ifstream ifs(args_->input);
  if (!ifs.is_open()) {
    std::cerr << "Input file cannot be opened!" << std::endl;
    exit(EXIT_FAILURE);
  }
  // 根据输入文件初始化词典
  dict_->readFromFile(ifs);
  ifs.close();

  // 存在预训练好的向量
  if (args_->pretrainedVectors.size() != 0) {
    loadVectors(args_->pretrainedVectors);
  } else {
  // 初始化输入层, 对于普通 word2vec，输入层就是一个词向量的查找表，
  // 所以它的大小为 nwords 行，dim 列（dim 为词向量的长度），但是 fastText 用了
  // word n-gram 作为输入，所以输入矩阵的大小为 (nwords + ngram 种类) * dim
  // 代码中，所有 word n-gram 都被 hash 到固定数目的 bucket 中，所以输入矩阵的大小为
  // (nwords + bucket 个数) * dim
  // shared_ptr<Matrix>: input_
    input_ = std::make_shared<Matrix>(dict_->nwords()+args_->bucket, args_->dim);
    input_->uniform(1.0 / args_->dim);
  }

  // 初始化输出层，输出层无论是用负采样，层次 softmax，还是普通 softmax，
  // 对于每种可能的输出，都有一个 dim 维的参数向量与之对应
  // 当 args_->model == model_name::sup 时，训练分类器，
  // 所以输出的种类是标签总数 dict_->nlabels()
  if (args_->model == model_name::sup) {
    // shared_ptr<Matrix>: output_
    output_ = std::make_shared<Matrix>(dict_->nlabels(), args_->dim);
  } else {
    // 否则训练的是词向量，输出种类就是词的种类 dict_->nwords()
    output_ = std::make_shared<Matrix>(dict_->nwords(), args_->dim);
  }
  output_->zero();
  startThreads();
  model_ = std::make_shared<Model>(input_, output_, args_, 0);

}

void FastText::startThreads() {
  start = clock();
  tokenCount = 0;
  if (args_->thread > 1) {
    std::vector<std::thread> threads;
    for (int32_t i = 0; i < args_->thread; i++) {
      // lambda表达式，[=] 以值的方式捕获所有的外部自动变量。
      threads.push_back(std::thread([=]() { trainThread(i); }));
    }
    for (auto it = threads.begin(); it != threads.end(); ++it) {
      it->join(); // 每个线程必须等待主线程
    }
  } else {
    trainThread(0);
  }
}

int FastText::getDimension() const {
    return args_->dim;
}

bool FastText::isQuant() const {
  return quant_;
}

}
