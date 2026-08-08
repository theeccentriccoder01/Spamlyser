const TelemetryLogger = require('../utils/telemetryLogger');

function errorMiddleware(err, req, res, next) {
  const correlationId = req.context?.correlationId;

  TelemetryLogger.error(correlationId, err.message, {
    stack: err.stack,
    path: req.originalUrl,
    method: req.method,
  });

  res.status(err.statusCode || 500).json({
    error: err.message || 'Internal Server Error',
    correlationId,
  });
}

module.exports = errorMiddleware;
