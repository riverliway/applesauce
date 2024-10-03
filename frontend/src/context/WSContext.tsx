import React, { ReactNode, useEffect, useState } from 'react'

import { localStageProd } from '../utils/localStageProd'
import { WsMessage, WsMessageType, WsMessageTypeMap } from './apiTypes'
import { ErrorPage } from '../components/Error/ErrorPage'

const WEBSOCKET_URL = localStageProd({
  local: 'ws://127.0.0.1:4000',
  prod: 'wss://apple-api.riverway.li'
})

type WsCallback<T extends WsMessageType> = (message: WsMessageTypeMap[T]) => void
interface WsCallbackLibrary {
  [key: string]: WsCallback<WsMessageType>[]
}

export interface WebsocketContext {
  /**
   * Messages the backend to start a new simulation
   */
  startSimulation: (simType: 'simple' | 'complex', width: number, height: number, numBots: number, numBaskets: number, seed: number) => void
  /**
   * Registers a callback for a specific message type
   * @param messageType - the type of message to register for
   * @param callback - the callback to call when a message of the specified type is received
   */
  register: <T extends WsMessageType>(messageType: T, callback: WsCallback<T>) => void
  /**
   * Removes this callback from the list of callbacks for the specified message type
   * @param messageType - the type of message to unregister for
   * @param callback - the callback to remove
   */
  unregister: <T extends WsMessageType>(messageType: T, callback: WsCallback<T>) => void
}

const InternalWebsocketContext = React.createContext(undefined as unknown as WebsocketContext)

export const useWebsocketContext = (): WebsocketContext => {
  const context = React.useContext(InternalWebsocketContext)
  if (context === undefined) {
    throw new Error('useWebsocketContext must be used in components that are wrapped around the WebsocketProvider.')
  }
  return context
}

export const WebsocketProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [ws] = useState(new WebSocket(WEBSOCKET_URL))
  const [networkError, setNetworkError] = useState(false)
  const [wsQueue, setWsQueue] = useState<string[]>([])
  const [callbackLibrary, setCallbackLibrary] = useState<WsCallbackLibrary>({})

  const register = <T extends WsMessageType>(messageType: T, callback: WsCallback<T>) => {
    const callbacks = callbackLibrary[messageType] ?? []
    setCallbackLibrary(cl => ({
      ...cl,
      [messageType]: [...callbacks, callback] as WsCallback<WsMessageType>[]
    }))
  }

  const unregister = <T extends WsMessageType>(messageType: T, callback: WsCallback<T>) => {
    const callbacks = callbackLibrary[messageType] ?? []
    setCallbackLibrary(cl => ({
      ...cl,
      [messageType]: callbacks.filter(cb => cb !== callback)
    }))
  }

  const sendWsMessage = (message: any) => {
    const msg = JSON.stringify(message)
    if (ws.readyState === ws.OPEN) {
      ws.send(msg)
    } else {
      // If the websocket is not open, add the message to the queue
      // to be sent when the websocket is open
      setWsQueue(q => [...q, msg])
    }
  }

  useEffect(() => {
    ws.onmessage = (event: MessageEvent) => {
      console.log('Received message:', event.data)
      try {
        const content = JSON.parse(event.data) as WsMessage
        const callbacks = callbackLibrary[content.type] ?? []
        callbacks.forEach(cb => cb(content))
      } catch (e) {
        console.error('Error parsing websocket message:')
        console.error(e)
      }
    }
  }, [callbackLibrary])

  useEffect(() => {
    ws.onopen = () => {
      wsQueue.forEach(sendWsMessage)
      setWsQueue([])
    }

    ws.onerror = () => {
      setNetworkError(true)
    }
  }, [wsQueue])

  const value = {
    register,
    unregister,
    startSimulation: (simType: 'simple' | 'complex', width: number, height: number, numBots: number, numBaskets: number, seed: number) => {
      sendWsMessage({
        type: 'start-simulation',
        params: {
          width,
          height,
          num_bots: numBots,
          num_baskets: numBaskets,
          seed,
          sim_type: simType
        }
      })
    }
  }

  if (networkError) {
    return <ErrorPage title='Network Error' description='There was an error connecting to the server, please try again later' buttonText='' />
  }

  return (
    <InternalWebsocketContext.Provider value={value}>
      {children}
    </InternalWebsocketContext.Provider>
  )
}
