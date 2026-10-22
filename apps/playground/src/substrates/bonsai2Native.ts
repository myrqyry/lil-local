/**
 * Optional PrismML llama.cpp native server connection.
 * Requires a user-started server with the actual Bonsai 2 2B GGUF loaded.
 * Never downloads weights, starts a server, or talks to non-loopback hosts.
 */
export interface BonsaiNativeConnection {
  baseUrl: string
  model: string
}

type ResponseLike = {
  ok: boolean
  status?: number
  json(): Promise<unknown>
}
type Fetcher = (url: string, init?: RequestInit) => Promise<ResponseLike>

const isBonsai2b = (model: string) => /bonsai[-_ ]?2[-_ ]?2b/i.test(model)

export function parseBonsaiNativePort(input: string): number {
  if (!/^\d{4,5}$/.test(input)) throw new Error('Choose a local server port between 1024 and 65535.')
  const port = Number(input)
  if (port < 1024 || port > 65535) throw new Error('Choose a local server port between 1024 and 65535.')
  return port
}

async function requestJson(
  fetcher: Fetcher,
  url: string,
  options: RequestInit,
): Promise<unknown> {
  const result = await fetcher(url, { ...options, redirect: 'error' })
  if (!result.ok) throw new Error('Native Bonsai server HTTP ' + (result.status ?? 'error'))
  return result.json()
}

/** Server identity check; never claims a generic OpenAI-compatible endpoint is Bonsai. */
export async function connectBonsaiNative(
  port: number,
  fetcher: Fetcher = fetch,
): Promise<BonsaiNativeConnection> {
  if (!Number.isInteger(port) || port < 1024 || port > 65535) {
    throw new Error('Invalid loopback server port')
  }
  const baseUrl = 'http://127.0.0.1:' + port
  const models = await requestJson(fetcher, baseUrl + '/v1/models', {
    method: 'GET', signal: AbortSignal.timeout(8_000),
  }) as { data?: { id?: unknown }[] }
  const model = models?.data?.map(entry => entry.id)
    .find((id): id is string => typeof id === 'string' && isBonsai2b(id))
  if (!model) throw new Error('The local server does not advertise a Bonsai 2 2B GGUF model.')
  return { baseUrl, model }
}

/** Bounded non-streaming request; native runtime executes outside the browser. */
export async function completeBonsaiNative(
  connection: BonsaiNativeConnection,
  prompt: string,
  fetcher: Fetcher = fetch,
): Promise<string> {
  if (!prompt.trim() || prompt.length > 4096) throw new Error('Enter a prompt (maximum 4096 characters).')
  if (!/^http:\/\/127\.0\.0\.1:\d{4,5}$/.test(connection.baseUrl) || !isBonsai2b(connection.model)) {
    throw new Error('Invalid Bonsai local connection identity')
  }
  const result = await requestJson(fetcher, connection.baseUrl + '/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type':'application/json' },
    body: JSON.stringify({
      model: connection.model,
      messages: [{ role:'user', content:prompt }],
      max_tokens:96,
      stream:false,
    }),
    signal: AbortSignal.timeout(45_000),
  }) as { model?: unknown; choices?: { message?: { content?: unknown } }[] }
  if (result?.model !== connection.model) throw new Error('The native completion response model differs from the connected model.')
  const content = result?.choices?.[0]?.message?.content
  if (typeof content !== 'string' || !content.trim()) {
    throw new Error('The native model returned no text; no output can be claimed.')
  }
  return content
}
