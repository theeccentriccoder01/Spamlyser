const express = require('express');
const telemetryContextMiddleware = require('./middleware/telemetryContext.middleware');
const errorMiddleware = require('./middleware/error.middleware');

const app = express();
app.use(express.json());
app.use(telemetryContextMiddleware);

app.get('/health', (req, res) => res.status(200).json({ status: 'ok' }));
app.get('/error', (req, res, next) => next(new Error('boom')));

app.use(errorMiddleware);

module.exports = app;
