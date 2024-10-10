import {
  MessageBody,
  SubscribeMessage,
  WebSocketGateway,
  WebSocketServer,
  WsResponse,
} from '@nestjs/websockets';
import { Socket } from 'dgram';
import { Observable } from 'rxjs';
import { Server } from 'socket.io';

@WebSocketGateway({
  cors: {
    origin: '*',
  },
})
export class EventsGateway {
  @WebSocketServer()
  server: Server;

  @SubscribeMessage('connected')
  connected(): Observable<WsResponse<number>> {
    return new Observable((observer) => {
      observer.next({
        event: 'connected',
        data: 200,
      });
    });
  }

  @SubscribeMessage('events')
  handleEvent(client: Socket, data: string): string {
    return data;
  }

  @SubscribeMessage('progress')
  async progress(@MessageBody() data: number): Promise<number> {
    this.server.emit('progress', data);
    return data;
  }
}
