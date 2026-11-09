import {describe, expect, it} from 'vitest'
import {canonicalWavBlob} from './gemma-audio'

function wav({sampleRate=16000, channels=1, bits=16, bytes=320}: {sampleRate?:number; channels?:number; bits?:number; bytes?:number}={}): File {
  const data = new Uint8Array(44+bytes)
  const view=new DataView(data.buffer)
  for(const [i,c] of [...'RIFF'].entries()) data[i]=c.charCodeAt(0)
  view.setUint32(4,36+bytes,true)
  for(const [i,c] of [...'WAVE'].entries()) data[8+i]=c.charCodeAt(0)
  for(const [i,c] of [...'fmt '].entries()) data[12+i]=c.charCodeAt(0)
  view.setUint32(16,16,true)
  view.setUint16(20,1,true)
  view.setUint16(22,channels,true)
  view.setUint32(24,sampleRate,true)
  view.setUint32(28,sampleRate*channels*bits/8,true)
  view.setUint16(32,channels*bits/8,true)
  view.setUint16(34,bits,true)
  for(const [i,c] of [...'data'].entries()) data[36+i]=c.charCodeAt(0)
  view.setUint32(40,bytes,true)
  return new File([data], 'test.wav',{type:'audio/wav'})
}

describe('real E2B native audio input',()=>{
  it('preserves canonical 16k mono 16-bit PCM WAV bytes as a Blob',async()=>{
    const data=wav()
    const blob=await canonicalWavBlob(data)
    expect(blob.type).toBe('audio/wav')
    expect(blob.size).toBe(data.size)
    expect(new Uint8Array(await blob.arrayBuffer())).toEqual(new Uint8Array(await data.arrayBuffer()))
  })
  it('refuses formats that the browser/native model path cannot accept',async()=>{
    await expect(canonicalWavBlob(wav({sampleRate:44100}))).rejects.toThrow(/16.?000|16 kHz/)
    await expect(canonicalWavBlob(wav({channels:2}))).rejects.toThrow(/mono/)
    await expect(canonicalWavBlob(wav({bits:8}))).rejects.toThrow(/16.bit/)
    await expect(canonicalWavBlob(new File([new Uint8Array(60)],'bad.wav',{type:'audio/wav'}))).rejects.toThrow(/WAV|RIFF/)
  })
})
