const request = require('supertest');
const app = require('../app');

describe('Telemetry context middleware', () => {
  it('attaches a correlation ID header to responses', async () => {
    const res = await request(app).get('/health');
    expect(res.statusCode).toBe(200);
    expect(res.headers['x-correlation-id']).toBeDefined();
  });

  it('propagates the same correlation ID through error responses', async () => {
    const res = await request(app).get('/error');
    expect(res.statusCode).toBe(500);
    expect(res.body.correlationId).toBeDefined();
    expect(res.headers['x-correlation-id']).toBe(res.body.correlationId);
  });

  it('respects an incoming x-correlation-id header', async () => {
    const res = await request(app)
      .get('/health')
      .set('x-correlation-id', 'test-fixed-id');
    expect(res.headers['x-correlation-id']).toBe('test-fixed-id');
  });
});
