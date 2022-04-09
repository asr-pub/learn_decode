# 0. build lang dir
#utils/prepare_lang.sh --position-dependent-phones false \
#    --num-nonsil-states 1 \
#	--num-sil-states 1 \
#    ./data/dict "<UNK>" data/lang/tmp data/lang

# 1. extract mfcc features
#./steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 1 data/toy_data/ log/ ./mfcc
#./steps/compute_cmvn_stats.sh data/toy_data/ log/ ./mfcc

# 2. train uni-gram language model
#ngram-count -text text -order 1 > text.txt
#ngram-count -read text.txt -order 1 -lm LM
#./utils/format_lm_sri.sh data/lang/ LM data/lang_new
#rm -rf data/lang 
#mv data/lang_new data/lang

# 3. train a monophone model
#./steps/train_mono.sh --nj 1 data/toy_data/ data/lang/ exp/mono

# 4. build HCLG graph
#./utils/mkgraph.sh data/lang/ ./exp/mono ./exp/mono/graph

# 5. decode
sdata=data/toy_data/
feats="ark,s,cs:apply-cmvn --utt2spk=ark:$sdata/utt2spk scp:$sdata/cmvn.scp scp:$sdata/feats.scp ark:- | add-deltas ark:- ark:- |"
gmm-decode-simple exp/mono/final.mdl exp/mono/graph/HCLG.fst "$feats" ark,t:decode.id ark,t:align.txt ark,t:lattice.txt
