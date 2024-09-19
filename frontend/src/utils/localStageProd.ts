import { removeFront } from './removeFront'

/**
 * @param config - the configuration for the local and prod environments
 * @returns the configuration for the current environment
 */
export const localStageProd = <L, P>(config: { local: L, prod: P }): L | P => {
  const origin = window.location.origin
  const url = cleanUrl(origin)

  if (url.includes('localhost') || url.includes('127.0.0.1') || url.includes('0.0.0.0')) return config.local

  return config.prod
}

const cleanUrl = (url: string): string => {
  const urlClean = url.endsWith('/') ? url.slice(0, -1) : url
  const cleanHttp = removeFront(urlClean, 'http://')
  const cleanHttps = removeFront(cleanHttp, 'https://')
  return removeFront(cleanHttps, 'www.')
}
