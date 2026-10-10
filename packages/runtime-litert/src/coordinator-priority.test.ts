import { describe, expect, it, vi } from 'vitest'
import { InferenceCoordinator } from './coordinator'

function gate() {
  let release!: () => void
  const promise = new Promise<void>((resolve) => { release = resolve })
  return { promise, release }
}

describe('InferenceCoordinator priority scheduling', () => {
  it('runs higher-priority waiting work before older ordinary work without preempting active work', async () => {
    const coordinator = new InferenceCoordinator()
    const order: string[] = []
    const activeGate = gate()

    const active = coordinator.run(async () => {
      order.push('active:start')
      await activeGate.promise
      order.push('active:end')
    }, undefined, 'active')

    await vi.waitFor(() => expect(coordinator.snapshot().activeLabel).toBe('active'))
    const ordinary = coordinator.run(async () => { order.push('ordinary') }, undefined, 'ordinary')
    const urgent = coordinator.run(async () => { order.push('urgent') }, undefined, 'urgent', { priority: 10 })
    const laterOrdinary = coordinator.run(async () => { order.push('later-ordinary') }, undefined, 'later-ordinary')

    expect(coordinator.snapshot().queued).toBe(3)
    expect(order).toEqual(['active:start'])
    activeGate.release()
    await Promise.all([active, ordinary, urgent, laterOrdinary])
    expect(order).toEqual(['active:start', 'active:end', 'urgent', 'ordinary', 'later-ordinary'])
    expect(coordinator.snapshot()).toEqual({ active: false, activeLabel: undefined, queued: 0 })
  })

  it('preserves FIFO for callers without priorities and for equal-priority jobs', async () => {
    const coordinator = new InferenceCoordinator()
    const seen: string[] = []
    const jobs = [
      coordinator.run(async () => { seen.push('one') }),
      coordinator.run(async () => { seen.push('two') }, undefined, 'two', { priority: 0 }),
      coordinator.run(async () => { seen.push('three') }),
    ]
    await Promise.all(jobs)
    expect(seen).toEqual(['one', 'two', 'three'])
  })

  it('rejects queued cancellation promptly even while a job is still active', async () => {
    const coordinator = new InferenceCoordinator()
    const activeGate = gate()
    const active = coordinator.run(() => activeGate.promise)
    await vi.waitFor(() => expect(coordinator.snapshot().active).toBe(true))

    const controller = new AbortController()
    const neverRun = vi.fn(async () => 'wrong')
    const waiting = coordinator.run(neverRun, controller.signal, 'urgent', { priority: 99 })
    expect(coordinator.snapshot().queued).toBe(1)
    controller.abort()

    await expect(waiting).rejects.toMatchObject({ name: 'AbortError' })
    expect(coordinator.snapshot().active).toBe(true)
    expect(coordinator.snapshot().queued).toBe(0)
    expect(neverRun).not.toHaveBeenCalled()
    activeGate.release()
    await active
  })

  it('continues after a priority task fails', async () => {
    const coordinator = new InferenceCoordinator()
    const seen: string[] = []
    const failed = coordinator.run(async () => {
      seen.push('failed')
      throw new Error('inference failed')
    }, undefined, 'failed', { priority: 5 })
    const completed = coordinator.run(async () => {
      seen.push('completed')
      return 'ok'
    })
    await expect(failed).rejects.toThrow('inference failed')
    await expect(completed).resolves.toBe('ok')
    expect(seen).toEqual(['failed', 'completed'])
    expect(coordinator.snapshot().active).toBe(false)
  })

  it('does not poison the queue with invalid priorities', async () => {
    const coordinator = new InferenceCoordinator()
    const task = vi.fn(async () => 'ok')
    await expect(coordinator.run(task, undefined, 'invalid', { priority: Number.NaN })).rejects.toBeInstanceOf(RangeError)
    await expect(coordinator.run(task, undefined, 'invalid', { priority: Infinity })).rejects.toBeInstanceOf(RangeError)
    expect(task).not.toHaveBeenCalled()
    expect(coordinator.snapshot().queued).toBe(0)
    await expect(coordinator.run(task)).resolves.toBe('ok')
  })

  it('reports queued/start/finish order and queue depth with priority switching', async () => {
    const coordinator = new InferenceCoordinator()
    const activeGate = gate()
    const events: string[] = []
    coordinator.on('queued', (event) => events.push(`queued:${event.label}:${event.queueDepth}`))
    coordinator.on('started', (event) => events.push(`started:${event.label}:${event.queueDepth}`))
    coordinator.on('finished', (event) => events.push(`finished:${event.label}:${event.queueDepth}`))
    const first = coordinator.run(() => activeGate.promise, undefined, 'first')
    await vi.waitFor(() => expect(coordinator.snapshot().activeLabel).toBe('first'))
    const old = coordinator.run(async () => undefined, undefined, 'old')
    const urgent = coordinator.run(async () => undefined, undefined, 'urgent', { priority: 3 })
    activeGate.release()
    await Promise.all([first, old, urgent])
    expect(events).toEqual([
      'queued:first:1', 'started:first:0',
      'queued:old:1', 'queued:urgent:2',
      'finished:first:2', 'started:urgent:1',
      'finished:urgent:1', 'started:old:0', 'finished:old:0',
    ])
  })
})
