import generateApiKey from 'generate-api-key';
import logger from '../config/logger';

const generatingApiKey = (): string => {
  const keys = generateApiKey({ method: 'base62' });
  return keys.toString();
};

export default generatingApiKey;
