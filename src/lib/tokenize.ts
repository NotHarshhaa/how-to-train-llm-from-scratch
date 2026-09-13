const GPT2_RE =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export function splitGpt2Pieces(text: string): string[] {
  if (!text) return [];
  return text.match(GPT2_RE) ?? [];
}

export function standInTokenId(piece: string): number {
  let h = 2166136261;
  for (let i = 0; i < piece.length; i++) {
    h ^= piece.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) % 50257;
}

export function tokenize(text: string) {
  const pieces = splitGpt2Pieces(text);
  return pieces.map((piece) => ({
    piece,
    id: standInTokenId(piece),
  }));
}
