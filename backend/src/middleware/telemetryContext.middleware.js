const crypto = require('crypto');
const TelemetryLogger = require('../utils/telemetryLogger');

function telemetryContextMiddleware(req, res, next) {
  const correlationId = req.headers['x-correlation-id'] || crypto.randomUUID();
  req.context = { correlationId, startTime: Date.now() };
  res.setHeader('x-correlation-id', correlationId);

  res.on('finish', () => {
    const duration = Date.now() - req.context.startTime;
    TelemetryLogger.info(correlationId, 'request completed', {
      method: req.method,
      path: req.originalUrl,
      statusCode: res.statusCode,
      durationMs: duration,
    });
  });

  next();
}

module.exports = telemetryContextMiddleware;
