
/**
 * Removes the front of a string if it exists
 * @param str - the string to remove the front of
 * @param front - the string to remove from the front of str
 * @returns str with front removed from the front
 */
export const removeFront = (str: string, front: string): string => {
  if (!str.startsWith(front)) return str
  return str.slice(front.length)
}

/**
 * Removes the back of a string if it exists
 * @param str - the string to remove the back of
 * @param back - the string to remove from the back of str
 * @returns str with back removed from the back
 */
export const removeBack = (str: string, back: string): string => {
  if (!str.endsWith(back)) return str
  return str.slice(0, str.length - back.length)
}
