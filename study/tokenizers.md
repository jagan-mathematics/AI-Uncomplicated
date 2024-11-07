**Finding 1: WordPiece Tokenization Improves NMT Performance**

NMT models often struggle with rare words or words not found in their vocabulary (OOV). This can significantly impact translation quality. A key finding is that using a WordPiece tokenizer helps address this issue. 

**Here's how WordPiece tokenization benefits NMT:**

* **Breaks down rare words:** By splitting infrequent words into smaller subword units, the model encounters these units more frequently during training. This improves its ability to translate them accurately.
* **Reduces vocabulary size:**  Instead of needing entries for every possible word, the model learns a smaller set of subword units that can be combined to represent most words. This can lead to more efficient training and potentially better generalization.

**Evidence:** This approach was used in a Google production NMT model and achieved state-of-the-art results at the time[1]. 

### Limitation of Word Tokenizer
Word-level tokenizers are relatively effective in processing European languages, where spaces provide clear word boundaries. However, this method is limited in languages like Chinese, which do not have clear word boundaries. Moreover, the flexible morphological inflections in language, the constant emergence of new words, and the
prevalence of spelling errors in corpora make it difficult for word-level vocabularies to generalize in practical applications.[3]


#### Lama2 Tokenizer train config[4]
```commandline
normalizer_spec {
  name: "identity"
  precompiled_charsmap: ""
  add_dummy_prefix: true
  remove_extra_whitespaces: false
  normalization_rule_tsv: ""
}

trainer_spec {
  input: "/large_experiments/theorem/datasets/MERGED/all.test1.merged"
  model_prefix: "spm_model_32k_200M_charcov099995_allowWSO__v2"
  model_type: BPE
  vocab_size: 32000
  self_test_sample_size: 0
  input_format: "text"
  character_coverage: 0.99995
  input_sentence_size: 200000000
  seed_sentencepiece_size: 1000000
  shrinking_factor: 0.75
  num_threads: 80
  num_sub_iterations: 2
  max_sentence_length: 4192
  shuffle_input_sentence: true
  max_sentencepiece_length: 16
  split_by_unicode_script: true
  split_by_whitespace: true
  split_by_number: true
  treat_whitespace_as_suffix: false
  split_digits: true
  allow_whitespace_only_pieces: true
  vocabulary_output_piece_score: true
  hard_vocab_limit: true
  use_all_vocab: false
  byte_fallback: true
  required_chars: ""
  unk_id: 0
  bos_id: 1
  eos_id: 2
  pad_id: -1
  unk_surface: " \342\201\207 "
  unk_piece: "<unk>"
  bos_piece: "<s>"
  eos_piece: "</s>"
  pad_piece: "<pad>"
  train_extremely_large_corpus: false
  enable_differential_privacy: false
  differential_privacy_noise_level: 0.0
  differential_privacy_clipping_threshold: 0
}
```



## Reference:
- [1] [wordpeice tokenizer in Google production **NMT** model](https://arxiv.org/pdf/1609.08144v2)
- [2] [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909)
- [3] [rethinking tokenizers](https://arxiv.org/pdf/2403.00417)
- [4] [Andrej Karpathy video](https://www.youtube.com/watch?v=zduSFxRajkE)
