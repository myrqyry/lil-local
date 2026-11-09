/**
 * Full Gemma 4 E2B uses LiteRT-LM's native audio encoder, whose proven
 * browser contract is 16 kHz mono 16-bit uncompressed PCM WAV, <= 8 MB.
 * Fail closed rather than sending unsupported compressed audio to WASM.
 */
export async function canonicalWavBlob(file: File): Promise<Blob> {
  if (file.size > 8_000_000) throw new Error('Gemma WAV audio exceeds the 8 MB native input limit')
  const bytes = new Uint8Array(await file.arrayBuffer())
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const chunk = (offset: number) => String.fromCharCode(...bytes.subarray(offset,offset+4))
  if (bytes.byteLength < 44 || chunk(0) !== 'RIFF' || chunk(8) !== 'WAVE') {
    throw new Error('Gemma audio must be a RIFF/WAVE file')
  }
  let fmt = false
  let samples = false
  for (let offset = 12; offset + 8 <= bytes.byteLength;) {
    const name = chunk(offset)
    const length = view.getUint32(offset + 4,true)
    const end = offset + 8 + length
    if (end > bytes.byteLength) throw new Error('Malformed WAV chunk length')
    if (name === 'fmt ') {
      if (length < 16) throw new Error('Malformed WAV format chunk')
      const codec = view.getUint16(offset+8,true)
      const channels = view.getUint16(offset+10,true)
      const rate = view.getUint32(offset+12,true)
      const bits = view.getUint16(offset+22,true)
      if (codec !== 1) throw new Error('Gemma audio requires uncompressed PCM WAV')
      if (channels !== 1) throw new Error('Gemma audio requires mono WAV')
      if (rate !== 16_000) throw new Error('Gemma audio requires 16 kHz (16000 Hz) WAV')
      if (bits !== 16) throw new Error('Gemma audio requires 16-bit PCM WAV')
      fmt = true
    }
    if (name === 'data' && length > 0) samples = true
    offset = end + (length & 1)
  }
  if (!fmt || !samples) throw new Error('Gemma audio requires WAV format and nonempty PCM data')
  return new Blob([bytes],{type:'audio/wav'})
}
