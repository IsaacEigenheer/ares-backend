import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  console.log('Nest is online!!');
  await app.listen(7001, '0.0.0.0');
}
bootstrap();
