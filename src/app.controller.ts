import {
  Controller,
  Get,
  Res,
  Post,
  UploadedFile,
  UseInterceptors,
  Query,
  Param, // Importado
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { spawn } from 'child_process';
import { diskStorage } from 'multer';
import { extname, join, resolve } from 'path'; // Importado 'resolve'
import { Response } from 'express';
import * as fs from 'fs';
import { EventsGateway } from './websocket/events.gateway';

@Controller('upload')
export class AppController {
  constructor(private readonly eventsGateway: EventsGateway) {}

  @Post()
  @UseInterceptors(
    FileInterceptor('file', {
      storage: diskStorage({
        destination: './uploads',
        filename: (req, file, callback) => {
          const uniqueSuffix = Date.now();
          const ext = extname(file.originalname);
          const filename = `output-${uniqueSuffix}${ext}`;
          callback(null, filename);
        },
      }),
    }),
  )
  uploadFile(
    @UploadedFile() file: any,
    @Res() res: Response,
    @Query() query: any,
  ) {
    let nomeDoArquivo: string;
    return new Promise((resolve, reject) => {
      // Garante que a pasta de screenshots exista na raiz do projeto
      fs.mkdirSync(join(__dirname, '..', 'debug_screenshots'), { recursive: true });

      const pythonProcess = spawn('python', [
        './python_backend/main.py',
        `../uploads/${file.filename}`,
        query.type,
        query.nPages,
      ]);

      const currentDirectory = __dirname;
      console.log('Diretório atual do controller:', currentDirectory);

      pythonProcess.stdout.on('data', (data) => {
        const message = data.toString().trim();
        const id = query.id;

        console.log(`Python stdout: ${message}`);

        if (message.startsWith('ExcelFinal')) {
          const parts = message.split(' ');
          if (parts.length > 1) {
            nomeDoArquivo = parts[1].replace(/[\r\n]+$/, '');
          }
        } else if (message.startsWith('DebugScreenshot')) {
          const parts = message.split(' ');
          if (parts.length > 1) {
            const screenshotFilename = parts[1].replace(/[\r\n]+$/, '');
            this.eventsGateway.server.emit('debug-screenshot-ready', {
              filename: screenshotFilename,
              id: id,
            });
          }
        } else if (message.startsWith('8')) {
          if (!nomeDoArquivo) {
            console.error('Nome do arquivo final não capturado!');
            if(!res.headersSent) res.status(500).send('Nome do arquivo não foi gerado.');
            return;
          }

          // Constrói o caminho para o arquivo final usando __dirname
          const filePath = join(currentDirectory, '..', 'python_backend', 'Excel', nomeDoArquivo);
          
          res.setHeader('Access-Control-Allow-Origin', '*');
          res.download(filePath, (err) => {
            if (err) {
              console.error('Erro ao baixar o arquivo:', err);
            }
            // Limpeza opcional do arquivo após o download
            fs.unlink(filePath, (unlinkErr) => {
              if (unlinkErr) console.error('Erro ao deletar arquivo final:', unlinkErr);
              else console.log('Arquivo final deletado com sucesso.');
            });
          });
        } else {
          const progress = parseInt(message);
          if (!isNaN(progress)) {
            this.eventsGateway.progress({ progress, id });
          }
        }
      });

      pythonProcess.stderr.on('data', (data) => {
        console.error(`Python stderr: ${data}`);
      });

      pythonProcess.on('close', (code) => {
        console.log(`Processo Python encerrado com código ${code}`);
        resolve(undefined);
      });
    });
  }

  @Get('debug-screenshot/:filename')
  getDebugScreenshot(
    @Param('filename') filename: string,
    @Res() res: Response,
  ) {
    // Usa __dirname para construir o caminho relativo e `resolve` para torná-lo absoluto
    const filePath = resolve(__dirname, '..', 'debug_screenshots', filename);
    if (fs.existsSync(filePath)) {
      res.sendFile(filePath);
    } else {
      res.status(404).send('Screenshot não encontrado.');
    }
  }
}