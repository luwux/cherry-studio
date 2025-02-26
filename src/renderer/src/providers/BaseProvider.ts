import { FOOTNOTE_PROMPT, REFERENCE_PROMPT } from '@renderer/config/prompts'
import { getLMStudioKeepAliveTime } from '@renderer/hooks/useLMStudio'
import { getOllamaKeepAliveTime } from '@renderer/hooks/useOllama'
import { getKnowledgeBaseReferences } from '@renderer/services/KnowledgeService'
import type {
  Assistant,
  GenerateImageParams,
  KnowledgeReference,
  Message,
  Model,
  Provider,
  Suggestion
} from '@renderer/types'
import { delay, isJSON, parseJSON } from '@renderer/utils'
import { addAbortController, removeAbortController } from '@renderer/utils/abortController'
import { formatApiHost } from '@renderer/utils/api'
import { TavilySearchResponse } from '@tavily/core'
import { t } from 'i18next'
import { isEmpty } from 'lodash'
import type OpenAI from 'openai'

import type { CompletionsParams } from '.'

// Character-by-character output controller class for smooth text display
export class SmoothTextOutput {
  private pendingText: string = ''
  private isRunning: boolean = false
  private charIntervalMs: number = 5 // Adjust this value to control output speed
  private pauseMillisec: number = 0
  private aborted: boolean = false
  private onChunkCallback: (text: string, reasoning?: string, metrics?: any, citations?: any) => void

  constructor(
    onChunkCallback: (text: string, reasoning?: string, metrics?: any, citations?: any) => void,
    charIntervalMs: number = 5
  ) {
    this.onChunkCallback = onChunkCallback
    this.charIntervalMs = charIntervalMs
  }

  public append(text: string = '', reasoning_content: string = '', metrics: any = {}, citations?: any): void {
    this.pendingText += text || ''

    // If not already running, start outputting characters
    if (!this.isRunning && this.pendingText.length > 0) {
      this.isRunning = true
      this.outputNextChar(reasoning_content, metrics, citations)
    }
  }

  public setPause(pause: number): void {
    this.pauseMillisec = pause
  }

  public abort(): void {
    this.aborted = true
    this.isRunning = false
  }

  private async outputNextChar(reasoning_content: string = '', metrics: any = {}, citations?: any): Promise<void> {
    if (this.aborted) return

    // If there's text to output
    if (this.pendingText.length > 0) {
      // Take the first character
      const char = this.pendingText.charAt(0)
      this.pendingText = this.pendingText.slice(1)

      // Send the character through the callback
      this.onChunkCallback(char, reasoning_content, metrics, citations)

      // Wait before outputting next character
      await new Promise((resolve) => setTimeout(resolve, this.charIntervalMs + this.pauseMillisec))

      // Continue with next character
      this.outputNextChar(reasoning_content, metrics, citations)
    } else {
      // No more text to output
      this.isRunning = false
    }
  }

  // Flush any remaining text immediately
  public flush(reasoning_content: string = '', metrics: any = {}, citations?: any): void {
    if (this.pendingText.length > 0) {
      this.onChunkCallback(this.pendingText, reasoning_content, metrics, citations)
      this.pendingText = ''
    }
    this.isRunning = false
  }
}

export default abstract class BaseProvider {
  protected provider: Provider
  protected host: string
  protected apiKey: string

  constructor(provider: Provider) {
    this.provider = provider
    this.host = this.getBaseURL()
    this.apiKey = this.getApiKey()
  }

  abstract completions({ messages, assistant, onChunk, onFilterMessages }: CompletionsParams): Promise<void>
  abstract translate(message: Message, assistant: Assistant, onResponse?: (text: string) => void): Promise<string>
  abstract summaries(messages: Message[], assistant: Assistant): Promise<string>
  abstract suggestions(messages: Message[], assistant: Assistant): Promise<Suggestion[]>
  abstract generateText({ prompt, content }: { prompt: string; content: string }): Promise<string>
  abstract check(model: Model): Promise<{ valid: boolean; error: Error | null }>
  abstract models(): Promise<OpenAI.Models.Model[]>
  abstract generateImage(params: GenerateImageParams): Promise<string[]>
  abstract getEmbeddingDimensions(model: Model): Promise<number>

  // Creates a smooth text output instance for character-by-character display
  protected createSmoothTextOutput(
    onChunk: (chunk: { text: string; reasoning_content?: string; usage?: any; metrics?: any; citations?: any }) => void
  ): SmoothTextOutput {
    return new SmoothTextOutput((text, reasoning_content, metrics, citations) => {
      onChunk({
        text,
        reasoning_content: reasoning_content || '',
        usage: metrics?.usage,
        metrics,
        citations
      })
    })
  }

  // Check if smooth character-by-character output should be enabled
  protected shouldUseSmoothOutput(): boolean {
    // Enable smooth output for all providers by default
    // This can be adjusted based on provider-specific requirements or user preferences
    return true
  }

  public getBaseURL(): string {
    const host = this.provider.apiHost
    return formatApiHost(host)
  }

  public getApiKey() {
    const keys = this.provider.apiKey.split(',').map((key) => key.trim())
    const keyName = `provider:${this.provider.id}:last_used_key`

    if (keys.length === 1) {
      return keys[0]
    }

    const lastUsedKey = window.keyv.get(keyName)
    if (!lastUsedKey) {
      window.keyv.set(keyName, keys[0])
      return keys[0]
    }

    const currentIndex = keys.indexOf(lastUsedKey)
    const nextIndex = (currentIndex + 1) % keys.length
    const nextKey = keys[nextIndex]
    window.keyv.set(keyName, nextKey)

    return nextKey
  }

  public defaultHeaders() {
    return {
      'X-Api-Key': this.apiKey
    }
  }

  public get keepAliveTime() {
    return this.provider.id === 'ollama'
      ? getOllamaKeepAliveTime()
      : this.provider.id === 'lmstudio'
        ? getLMStudioKeepAliveTime()
        : undefined
  }

  public async fakeCompletions({ onChunk }: CompletionsParams) {
    // Create smooth output instance
    const smoothOutput = this.shouldUseSmoothOutput() ? this.createSmoothTextOutput(onChunk) : null

    for (let i = 0; i < 100; i++) {
      await delay(0.01)
      const text = i + '\n'
      const metrics = { completion_tokens: 0, prompt_tokens: 0, total_tokens: 0 }

      if (smoothOutput) {
        smoothOutput.append(text, '', { usage: metrics })
      } else {
        onChunk({ text, usage: metrics })
      }
    }

    // Flush any remaining text
    if (smoothOutput) {
      smoothOutput.flush()
    }
  }

  public async getMessageContent(message: Message) {
    const webSearchReferences = await this.getWebSearchReferences(message)

    if (!isEmpty(webSearchReferences)) {
      const referenceContent = `\`\`\`json\n${JSON.stringify(webSearchReferences, null, 2)}\n\`\`\``
      return REFERENCE_PROMPT.replace('{question}', message.content).replace('{references}', referenceContent)
    }

    const knowledgeReferences = await getKnowledgeBaseReferences(message)

    if (!isEmpty(message.knowledgeBaseIds) && isEmpty(knowledgeReferences)) {
      window.message.info({ content: t('knowledge.no_match'), key: 'knowledge-base-no-match-info' })
    }

    if (!isEmpty(knowledgeReferences)) {
      const referenceContent = `\`\`\`json\n${JSON.stringify(knowledgeReferences, null, 2)}\n\`\`\``
      return FOOTNOTE_PROMPT.replace('{question}', message.content).replace('{references}', referenceContent)
    }

    return message.content
  }

  private async getWebSearchReferences(message: Message) {
    const webSearch: TavilySearchResponse = window.keyv.get(`web-search-${message.id}`)

    if (webSearch) {
      return webSearch.results.map(
        (result, index) =>
          ({
            id: index + 1,
            content: result.content,
            sourceUrl: result.url,
            type: 'url'
          }) as KnowledgeReference
      )
    }

    return []
  }

  protected getCustomParameters(assistant: Assistant) {
    return (
      assistant?.settings?.customParameters?.reduce((acc, param) => {
        if (!param.name?.trim()) {
          return acc
        }
        if (param.type === 'json') {
          const value = param.value as string
          if (value === 'undefined') {
            return { ...acc, [param.name]: undefined }
          }
          return { ...acc, [param.name]: isJSON(value) ? parseJSON(value) : value }
        }
        return {
          ...acc,
          [param.name]: param.value
        }
      }, {}) || {}
    )
  }

  protected createAbortController(messageId?: string) {
    const abortController = new AbortController()

    if (messageId) {
      addAbortController(messageId, () => abortController.abort())
    }

    return {
      abortController,
      cleanup: () => {
        if (messageId) {
          removeAbortController(messageId)
        }
      }
    }
  }
}
