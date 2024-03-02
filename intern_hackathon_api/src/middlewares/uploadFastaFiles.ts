import multer, { StorageEngine } from 'multer';
import { Request, Response, NextFunction } from 'express';
import httpStatus from 'http-status';
import { AuthRequest } from '../types/authRequest';
import fs from 'fs';
import path from 'path';

// Set up Multer storage and file filter
const storage: StorageEngine = multer.memoryStorage(); // You can customize the storage as needed

const fileFilter = (req: Request, file: Express.Multer.File, cb: multer.FileFilterCallback) => {
  if (file.mimetype === 'text/plain') {
    cb(null, true);
  } else {
    cb(new Error('Only text/plain files are allowed!'));
  }
};

// Set up Multer upload middleware
const upload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: {
    fileSize: 1024 * 1024 * 10 // 10 MB (adjust as needed)
  }
});

const isFastaFormat = (content: string): boolean => {
  const fastaRegex = /^[^\S\n]*>[^\s>].*(?:\n[^\S\n]*[AGTC]+)+$/;
  return fastaRegex.test(content);
};

// Middleware function to handle FASTA file upload
const uploadFasta = (req: AuthRequest, res: Response, next: NextFunction) => {
  const nameFileUserDate = `${req.user.id}-${Date.now()}.fasta`;
  const singleUpload = upload.single(nameFileUserDate); // 'fastaFile' is the name attribute in the form

  singleUpload(req, res, async (err: any) => {
    if (err instanceof multer.MulterError) {
      return res.status(httpStatus.BAD_REQUEST).json({ error: err.message });
    } else if (err) {
      return res.status(httpStatus.INTERNAL_SERVER_ERROR).json({ error: 'Internal Server Error' });
    }

    // Access the file in the middleware using req.file
    if (!req.file) {
      return res.status(httpStatus.BAD_REQUEST).json({ error: 'No file provided' });
    }

    // Check if the file is in FASTA format
    const content = req.file.buffer.toString();
    if (!isFastaFormat(content)) {
      return res.status(httpStatus.BAD_REQUEST).json({ error: 'Invalid FASTA format' });
    }

    // Save the file to /static/fastaFiles
    const filePath = path.join(__dirname, '../static/fastaFiles', req.file.originalname);

    try {
      await fs.promises.writeFile(filePath, req.file.buffer);
      console.log('File saved successfully:', filePath);
    } catch (writeError) {
      console.error('Error saving file:', writeError);
      return res.status(httpStatus.INTERNAL_SERVER_ERROR).json({ error: 'Error saving the file' });
    }

    // You can process the file here or pass it to the next middleware
    (req as any).fastaFileName = nameFileUserDate;
    (req as any).fastaFilePath = req.file;
    next();
  });
};

export default uploadFasta;
