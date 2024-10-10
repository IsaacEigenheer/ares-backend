import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { EventsModule } from './websocket/events.module';
import { EventsGateway } from './websocket/events.gateway';

@Module({
  imports: [EventsModule],
  controllers: [AppController],
  providers: [AppService, EventsGateway],
})
export class AppModule {}
