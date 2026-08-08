class TelemetryLogger {
  static log({ level = 'info', correlationId, message, meta = {} }) {
    const entry = {
      timestamp: new Date().toISOString(),
      level,
      correlationId,
      message,
      ...meta,
    };
    const line = JSON.stringify(entry);
    if (level === 'error') console.error(line);
    else console.log(line);
    return entry;
  }

  static info(correlationId, message, meta) {
    return this.log({ level: 'info', correlationId, message, meta });
  }

  static error(correlationId, message, meta) {
    return this.log({ level: 'error', correlationId, message, meta });
  }
}

module.exports = TelemetryLogger;
