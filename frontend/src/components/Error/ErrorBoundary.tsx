import React, { ErrorInfo, ReactNode } from 'react'

export interface ErrorBoundaryProps {
  catch: (error: Error, errorInfo: ErrorInfo) => void
  alternate?: ReactNode
  children?: ReactNode
}

/**
 * The React equivalent of a try/catch
 * @prop `catch` - a callback to execute when the error occurs
 * @prop `alternate` - the component to render if an error occurs
 * @prop `children` - the component that can potentially throw an error
 */
export class ErrorBoundary extends React.Component<ErrorBoundaryProps, { thrown: boolean }> {
  constructor (props: ErrorBoundaryProps) {
    super(props)
    this.state = { thrown: false }
  }

  componentDidCatch (error: Error, errorInfo: ErrorInfo): void {
    this.setState({ thrown: true })
    this.props.catch(error, errorInfo)
  }

  render (): ReactNode {
    if (this.state.thrown) {
      return this.props.alternate ?? <></>
    }

    return this.props.children
  }
}
