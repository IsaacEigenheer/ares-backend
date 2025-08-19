import {
  Controller,
  Get,
  Res,
  Post,
  UseInterceptors,
  Query,
  UploadedFiles,
} from '@nestjs/common';
import { FileFieldsInterceptor } from '@nestjs/platform-express';
import { spawn } from 'child_process';
import { diskStorage } from 'multer';
import { extname, join } from 'path';
import { Response } from 'express';
import * as fs from 'fs';
import { EventsGateway } from './websocket/events.gateway';

@Controller('upload')
export class AppController {
  constructor(private readonly eventsGateway: EventsGateway) {}

  @Post()
  @UseInterceptors(
    FileFieldsInterceptor(
      [
        { name: 'file1', maxCount: 1 },
        { name: 'file2', maxCount: 1 },
      ],
      {
        storage: diskStorage({
          destination: './uploads',
          filename: (req, file, callback) => {
            const uniqueSuffix = Date.now();
            const ext = extname(file.originalname);
            const filename = `output${uniqueSuffix}${ext}`;
            callback(null, filename);
          },
        }),
      },
    ),
  )
  uploadFile(
    @UploadedFiles()
    files: { file1?: Express.Multer.File[]; file2?: Express.Multer.File[] },
    @Res() res: Response,
    @Query() query: any,
  ) {
    return new Promise((resolve, reject) => {
      const file1 = files?.file1?.[0];
      const file2 = files?.file2?.[0];

      if (!file1) {
        res.status(400).send('file1 is required');
        return reject('file1 is required');
      }

      // Decide args based on revision flag
      const isRevision = query.revision === 'true' || query.revision === true;

      let pythonArgs: string[] = [];

      if (isRevision) {
        if (!file2) {
          res.status(400).send('file2 is required for revision mode');
          return reject('file2 missing in revision mode');
        }

        // python main expects: <path1> <path2> <client> <page1> <page2>
        pythonArgs = [
          './python_backend/main.py',
          `../uploads/${file1.filename}`,
          `../uploads/${file2.filename}`,
          query.type,
          query.page1?.toString() ?? '',
          query.page2?.toString() ?? '',
        ];
      } else {
        // backward compatible: <path> <client> <page>
        pythonArgs = [
          './python_backend/main.py',
          `../uploads/${file1.filename}`,
          query.type,
          query.page ?? query.nPages ?? '',
        ];
      }

      const pythonProcess = spawn('python', pythonArgs);

      const currentDirectory = __dirname;
      console.log('Diretório atual do controller:', currentDirectory);
      console.log('Chamando python com args:', pythonArgs);

      let nomeDoArquivo: string;

      pythonProcess.stdout.on('data', (data) => {
        const str = data.toString();
        const progress = parseInt(str); // se stdout enviar apenas número como progresso
        const id = query.id;

        if (!isNaN(progress)) {
          this.eventsGateway.progress({ progress, id });
        }

        console.log(`PYTHON STDOUT: ${str}`);

        // detectar linha com nome final do excel (mantive a sua convenção "ExcelFinal <filename>")
        if (str.startsWith('ExcelFinal')) {
          const parts = str.split(' ');
          if (parts.length > 1) {
            nomeDoArquivo = parts[1].replace(/[\r\n]+$/, '');
            console.log('Nome do arquivo final detectado:', nomeDoArquivo);
          } else {
            console.log('Formato inesperado da string ExcelFinal:', str);
          }
        }

        // quando python sinaliza que pode fazer o download (no seu código era '8' como exemplo)
        // mantive sua checagem original: se começa com '8' -> envia o arquivo
        if (str.startsWith('8')) {
          if (!nomeDoArquivo) {
            console.error('Tentativa de download mas nomeDoArquivo não definido.');
            res.status(500).send('Output file not found');
            return;
          }

          const filePath: string = join(
            currentDirectory,
            `../python_backend/${nomeDoArquivo}`,
          );

          res.setHeader('Access-Control-Allow-Origin', '*');

          return res.download(filePath, (err) => {
            if (err) {
              console.error('Error downloading file:', err);
              res.status(500).send('Error downloading file');
              return reject(err);
            } else {
              // tenta deletar o arquivo gerado no python_backend
              fs.unlink(filePath, (errUnlink) => {
                if (errUnlink) {
                  console.error('Error deleting file:', errUnlink);
                } else {
                  console.log('File deleted successfully');
                }
              });
              return resolve(true);
            }
          });
        }
      });

      pythonProcess.stderr.on('data', (data) => {
        console.error(`PYTHON STDERR: ${data.toString()}`);
      });

      pythonProcess.on('close', (code) => {
        console.log(`python process exited with code ${code}`);
        // caso o download já tenha sido feito, promise já foi resolvida; senão, resolve para não deixar pendente
        resolve(true);
      });

      pythonProcess.on('error', (err) => {
        console.error('Failed to start python process:', err);
        reject(err);
      });
    });
  }

  @Get('download-excel')
  downloadExcel(@Res() res: Response) {
    // endpoint reservado se precisar separar a lógica de download depois.
    res.status(204).send();
  }
}
