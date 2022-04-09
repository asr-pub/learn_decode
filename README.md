# 搞懂 Kaldi 解码器之 Simple Decoder

### 背景知识

语音识别中的解码使用的是令牌传递（ Token Passing） 实现的 Viterbi 算法，在解码图（HCLG）和声学特征的条件下，进行令牌传递。在实际使用中，解码图往往非常庞大，这时候需要在解码的过程中，进行剪枝操作，将得分比较低的 token 予以删除。每输入一帧声学特征，令牌就向所有可能的路径传递一次，当执行到最后一帧时，令牌传递结束，此时查看所有终止状态上的令牌，取最优的一个或多个令牌。

### 编译运行

学习解码器的代码，最好自己修改代码编译运行一下。解码所需要的 声学模型、HCLG 和声学特征，我都已经准备好了，你只需要运行 `sh run.sh` 即可。下面是我准备的环境，其中使用的两个音频文件来自于 [Kaldi解码原理](https://www.sikiedu.com/my/course/940)。

1. 在 Linux 环境中编译安装 Kaldi，我这边是在 Mac 上面的 Ubuntu 虚拟机中安装的，除了速度慢点，其他没有什么问题
2. 将 [learn_decode](https://github.com/asr-pub/learn_decode) 下载下来，放置于 `kaldi/egs/` 目录下
3. 进入 `kaldi/egs/learn_decode/s5` 目录下，执行 `sh run.sh` 即可看到解码输出，`run.sh` 中还有其他的命令，如 `lang` 目录准备、特征提取、语言模型训练等

如果修改了解码器相关的代码，那么也修改一下目录下面的 `Makefile` 文件，只编译修改的文件就行。

### 相关参数

- prev_toks_: 数据类型是 dict，key 是 state，value 是 state 上面的 token，该 dict 存储上一帧的 <state, token>

- cur_toks_: 同上，该 dict 存储当前帧的 <state, token>
- num_frames_ready: 含义是有多少帧数据可以提供给 decodable 对象使用，由于我们这边是非流式解码，直接输入所有的语音帧，所以 num_frames_ready     这里的值就是输入的音频帧数

- num_frames_decoded_: 当前已解码的帧数，当 `num_frames_decoded_ >= num_frames_ready` 就停止解码
- acoustic_cost: 声学代价，实现方式如下

```c++
// decodable 对象中似然函数的实现
// scale_ 是声学分数缩放因子，默认是 0.1
virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
  return scale_*LogLikelihoodZeroBased(frame,
                                       trans_model_.TransitionIdToPdf(tid));
}

// 解码中 acoustic_cost 实现，加了个负号，表示值越小似然越大
BaseFloat acoustic_cost = -decodable->LogLikelihood(frame, arc.ilabel);
```

### 源码解析

下述代码只摘取了部分比较核心的源码，大家可以照着源码路径完整的去看一遍。

`kaldi/src/gmmbin/gmm-decode-simple.cc`

```c++
int main(int argc, char *argv[]) {
  // 允许解码没有走到最终节点
  bool allow_partial = true; 
  // 因为声学模型的分数和语言模型的分数范围不一样，所以需要对声学模型进行缩放
	BaseFloat acoustic_scale = 0.1;
  // 剪枝相关的参数，beam 越大，解码速度越慢，准确率越高
  BaseFloat beam = 16.0;
  // 该类中有 transition-id 到 pdf 到映射
  TransitionModel trans_model;
  // 声学模型，此时是 GMM 模型
  AmDiagGmm am_gmm;
  // 读取 HCLG.fst 文件
  Fst<StdArc> *decode_fst = ReadFstKaldiGeneric(fst_in_filename);
  
  BaseFloat tot_like = 0.0;
  // 解码的总帧数，比如有 10 条音频，每条音频 100 帧，那么 frame_count 最后的值为 1000
  kaldi::int64 frame_count = 0;
  // 解码成功、失败的音频数量
  int num_success = 0, num_fail = 0;
  // 解码的核心，输入 HCLG.fst 和 剪枝参数
  SimpleDecoder decoder(*decode_fst, beam);
  
  // 开始对每条音频进行解码了
  for (; !feature_reader.Done(); feature_reader.Next()) {
    
    // decodable 对象，这个对象可以获得声学分数（输入特征和 pdf）
  	DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
                                             acoustic_scale);
    // 开始解码，核心代码
    decoder.Decode(&gmm_decodable);
  }
}
```

`kaldi/src/decoder/simple-decoder.cc`

```c++
// 解码核心方法
bool SimpleDecoder::Decode(DecodableInterface *decodable) {
  InitDecoding();
  AdvanceDecoding(decodable);
  return (!cur_toks_.empty());
}

// 解码初始化方法
void SimpleDecoder::InitDecoding() {
  // clean up from last time:
  ClearToks(cur_toks_);
  ClearToks(prev_toks_);
  // initialize decoding:
  // start_state 在我们的 HCLG 中值为 5
  StateId start_state = fst_.Start();
  // fst::kNoStateId 的值为 -1
  KALDI_ASSERT(start_state != fst::kNoStateId);
  // 需要注意的是 StdWeight::One() 的值为 0
  StdArc dummy_arc(0, 0, StdWeight::One(), start_state);
  // 在起始节点上面创建 Token，输入的三个参数分别是 arc，acoustic_cost 和 *prev
  cur_toks_[start_state] = new Token(dummy_arc, 0.0, NULL);
  num_frames_decoded_ = 0;
  // 扩展虚边，在我们的解码图中，从开始节点出来之后没有虚边
  ProcessNonemitting();
}

// decodable 对象可以获得声学打分，如果 max_num_frames >= 0，那么解码最多不能超过 max_num_frames 帧
// max_num_frames 默认值为 -1
void SimpleDecoder::AdvanceDecoding(DecodableInterface *decodable,
                                      int32 max_num_frames) {
  KALDI_ASSERT(num_frames_decoded_ >= 0 &&
               "You must call InitDecoding() before AdvanceDecoding()");
  // num_frames_ready 的含义是有多少帧数据可以提供给 decodable 对象使用的
  // 由于我们这边是非流式解码，直接输入所有的语音帧，所以 num_frames_ready
  // 这里的值就是输入的音频帧数
  int32 num_frames_ready = decodable->NumFramesReady();
  // num_frames_ready must be >= num_frames_decoded, or else
  // the number of frames ready must have decreased (which doesn't
  // make sense) or the decodable object changed between calls
  // (which isn't allowed).
  KALDI_ASSERT(num_frames_ready >= num_frames_decoded_);
  int32 target_frames_decoded = num_frames_ready;
  if (max_num_frames >= 0)
    target_frames_decoded = std::min(target_frames_decoded,
                                     num_frames_decoded_ + max_num_frames);
  // 逐帧进行解码
  while (num_frames_decoded_ < target_frames_decoded) {
    // note: ProcessEmitting() increments num_frames_decoded_
    // 将 cur_toks 的值赋给 prev_toks_，然后对 prev_toks 中的 token 进行扩展，
    // 扩展的 token 又放在 cur_toks 中
    ClearToks(prev_toks_);
    // 执行完这句代码后，cur_toks_ 中的数据会被全部清空
    cur_toks_.swap(prev_toks_);
    // ProcessEmitting 和 ProcessNonemitting 就是解码核心中的核心了
    ProcessEmitting(decodable);
    ProcessNonemitting();
    PruneToks(beam_, &cur_toks_);
  }
}

// 扩展实边
void SimpleDecoder::ProcessEmitting(DecodableInterface *decodable) {
  int32 frame = num_frames_decoded_;
  // Processes emitting arcs for one frame.  Propagates from
  // prev_toks_ to cur_toks_.
  // 初始化剪枝参数为无穷大
  double cutoff = std::numeric_limits<BaseFloat>::infinity();
  // 遍历所有 当前 节点的 Token
  for (unordered_map<StateId, Token*>::iterator iter = prev_toks_.begin();
       iter != prev_toks_.end();
       ++iter) {
    // HCLG 图中的节点
    StateId state = iter->first;
    // 节点里面的 token
    Token *tok = iter->second;
    KALDI_ASSERT(state == tok->arc_.nextstate);
    // 遍历当前 state 所有的实边，fst_ 是当前的 HCLG 解码图
    for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst_, state);
         !aiter.Done();
         aiter.Next()) {
      const StdArc &arc = aiter.Value();
      // ilabel != 0 的边都是实边了
      if (arc.ilabel != 0) {  // propagate..
        // 计算当前 frame 在 transition-id 等于 arc.ilabel 时的声学似然，似然越大越好
        // 但是我们这边需要的是代价，需要越小越好，所有前面加了个负号
        BaseFloat acoustic_cost = -decodable->LogLikelihood(frame, arc.ilabel);
        double total_cost = tok->cost_ + arc.weight.Value() + acoustic_cost;

        if (total_cost >= cutoff) continue;
        if (total_cost + beam_  < cutoff)
          cutoff = total_cost + beam_;
        // 创建该扩展实边的 token
        Token *new_tok = new Token(arc, acoustic_cost, tok);
        unordered_map<StateId, Token*>::iterator find_iter
            = cur_toks_.find(arc.nextstate);
        
        // 如果当前扩展的节点上面已经有 token 了，那么需要和最新创建的 token 进行比较
        // 谁的 cost 比较小，就保留谁的
        if (find_iter == cur_toks_.end()) {
          cur_toks_[arc.nextstate] = new_tok;
        } else {
          if ( *(find_iter->second) < *new_tok ) {
            Token::TokenDelete(find_iter->second);
            find_iter->second = new_tok;
          } else {
            Token::TokenDelete(new_tok);
          }
        }
      }
    }
  }
  num_frames_decoded_++;
}

// 扩展虚边，与扩展实边在逻辑上基本上是一模一样的
void SimpleDecoder::ProcessNonemitting() {
  // Processes nonemitting arcs for one frame.  Propagates within
  // cur_toks_.
  std::vector<StateId> queue;
  double infinity = std::numeric_limits<double>::infinity();
  double best_cost = infinity;
  for (unordered_map<StateId, Token*>::iterator iter = cur_toks_.begin();
       iter != cur_toks_.end();
       ++iter) {
    queue.push_back(iter->first);
    best_cost = std::min(best_cost, iter->second->cost_);
  }
  // best_cost 是 cur_toks 里面的最小值
  double cutoff = best_cost + beam_;

  while (!queue.empty()) {
    StateId state = queue.back();
    queue.pop_back();
    Token *tok = cur_toks_[state];
    KALDI_ASSERT(tok != NULL && state == tok->arc_.nextstate);
    for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst_, state);
         !aiter.Done();
         aiter.Next()) {
      const StdArc &arc = aiter.Value();
      // 空边的 transition-id 为 0
      if (arc.ilabel == 0) {  // propagate nonemitting only...
        // 空边扩展不消耗音频帧
        const BaseFloat acoustic_cost = 0.0;
        Token *new_tok = new Token(arc, acoustic_cost, tok);
        if (new_tok->cost_ > cutoff) {
          Token::TokenDelete(new_tok);
        } else {
          unordered_map<StateId, Token*>::iterator find_iter
              = cur_toks_.find(arc.nextstate);
          if (find_iter == cur_toks_.end()) {
            cur_toks_[arc.nextstate] = new_tok;
            queue.push_back(arc.nextstate);
          } else {
            if ( *(find_iter->second) < *new_tok ) {
              Token::TokenDelete(find_iter->second);
              find_iter->second = new_tok;
              queue.push_back(arc.nextstate);
            } else {
              Token::TokenDelete(new_tok);
            }
          }
        }
      }
    }
  }
}

// 剪枝操作，只保留 cost 符合 （0，best_cost+beam） 的 token
void SimpleDecoder::PruneToks(BaseFloat beam, unordered_map<StateId, Token*> *toks) {
  if (toks->empty()) {
    KALDI_VLOG(2) <<  "No tokens to prune.\n";
    return;
  }
  double best_cost = std::numeric_limits<double>::infinity();
  // 获取最小的代价
  for (unordered_map<StateId, Token*>::iterator iter = toks->begin();
       iter != toks->end(); ++iter)
    best_cost = std::min(best_cost, iter->second->cost_);
  std::vector<StateId> retained;
  double cutoff = best_cost + beam;
  for (unordered_map<StateId, Token*>::iterator iter = toks->begin();
       iter != toks->end(); ++iter) {
    if (iter->second->cost_ < cutoff)
      // 将符合条件的 state 放入 retained 数据中
      retained.push_back(iter->first);
    else
      Token::TokenDelete(iter->second);
  }
  unordered_map<StateId, Token*> tmp;
  for (size_t i = 0; i < retained.size(); i++) {
    tmp[retained[i]] = (*toks)[retained[i]];
  }
  KALDI_VLOG(2) <<  "Pruned to " << (retained.size()) << " toks.\n";
  tmp.swap(*toks);
}
```

### 待解问题

1. 为什么会形成空边？

### 相关链接

[Kaldi解码原理 - 按行分析Simple-Decoder](https://www.sikiedu.com/my/course/940)

[Kaldi中的decoder(一）- 基础和viterbi解码](http://placebokkk.github.io/kaldi/2019/07/31/asr-kaldi-decoder1.html)

[token passing 算法动画](https://speech.zone/token-passing/)

