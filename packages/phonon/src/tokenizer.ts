/**
 * Minimal Unigram tokenizer for the phonon-31m-en ONNX model.
 *
 * Reads the Hugging Face `tokenizer.json` (Unigram model, Metaspace pre-tokenizer,
 * ByteFallback decoder) and encodes text into token IDs suitable for the flow_lm
 * graph. This is a self-contained implementation with no dependency on
 * @huggingface/transformers, keeping the package lightweight.
 */
export class PhononTokenizer {
  private readonly tokenToId = new Map<string, number>();
  private readonly idToToken = new Map<number, string>();
  private readonly tokenToScore = new Map<string, number>();

  private readonly unkId: number;
  private readonly bosId: number;
  private readonly eosId: number;
  private readonly padId: number;

  constructor(data: Record<string, unknown>) {
    const addedTokens: Array<{ id: number; content: string }> =
      (data.added_tokens as Array<{ id: number; content: string }>) ?? [];

    // Added tokens carry explicit IDs and take precedence.
    for (const at of addedTokens) {
      this.tokenToId.set(at.content, at.id);
      this.idToToken.set(at.id, at.content);
    }

    const model = data.model as Record<string, unknown> ?? {};
    const vocab: Array<[string, number]> = (model.vocab as Array<[string, number]>) ?? [];

    let nextId = 0;
    for (const at of addedTokens) {
      if (at.id >= nextId) nextId = at.id + 1;
    }

    for (const [token, score] of vocab) {
      if (!this.tokenToId.has(token)) {
        this.tokenToId.set(token, nextId);
        this.idToToken.set(nextId, token);
        nextId++;
      }
      this.tokenToScore.set(token, score);
    }

    this.unkId = (model.unk_id as number) ?? 0;

    const bos = addedTokens.find((t) => t.content === '');
    this.bosId = bos ? bos.id : 1;
    const eos = addedTokens.find((t) => t.content === '');
    this.eosId = eos ? eos.id : 2;
    const pad = addedTokens.find((t) => t.content === '<pad>');
    this.padId = pad ? pad.id : 3;
  }

  /** Encode text into token IDs, wrapped with BOS and EOS. */
  encode(text: string): number[] {
    const normalized = this.normalize(text);
    const words = this.preTokenize(normalized);

    const ids: number[] = [this.bosId];
    for (const word of words) {
      if (word.length === 0) continue;
      const tokens = this.segmentUnigram(word);
      for (const tok of tokens) {
        ids.push(this.tokenToId.get(tok) ?? this.unkId);
      }
    }
    ids.push(this.eosId);
    return ids;
  }

  private normalize(text: string): string {
    return text.replace(/\s+/g, ' ').trim();
  }

  /**
   * Metaspace pre-tokenization: replace spaces with U+2581 (▁) and split on it.
   * Each resulting word retains its leading ▁ marker.
   */
  private preTokenize(text: string): string[] {
    const metaspace = text.replace(/ /g, '\u2581');
    return metaspace.split('\u2581');
  }

  /**
   * Unigram segmentation via dynamic programming: finds the highest-scoring
   * tokenization of a word using the vocab scores, with byte-level fallback for
   * characters that appear in no vocab entry.
   */
  private segmentUnigram(word: string): string[] {
    const n = word.length;
    const dp: number[] = new Array(n + 1).fill(-Infinity);
    const back: number[] = new Array(n + 1).fill(-1);
    const backToken: string[] = new Array(n + 1).fill('');
    dp[0] = 0;

    for (let i = 1; i <= n; i++) {
      for (let j = 0; j < i; j++) {
        if (dp[j] === -Infinity) continue;
        const sub = word.slice(j, i);
        const score = this.tokenToScore.get(sub);
        if (score === undefined) continue;
        const cand = dp[j] + score;
        if (cand > dp[i]) {
          dp[i] = cand;
          back[i] = j;
          backToken[i] = sub;
        }
      }
      // Byte fallback: no vocab entry covers this position; emit the single byte.
      if (dp[i] === -Infinity) {
        const ch = word[i - 1];
        dp[i] = dp[i - 1] + (this.tokenToScore.get(ch) ?? -20);
        back[i] = i - 1;
        backToken[i] = ch;
      }
    }

    const tokens: string[] = [];
    let i = n;
    while (i > 0) {
      tokens.push(backToken[i]);
      i = back[i];
    }
    return tokens.reverse();
  }

  getUnkId(): number {
    return this.unkId;
  }

  getBosId(): number {
    return this.bosId;
  }

  getEosId(): number {
    return this.eosId;
  }
}
