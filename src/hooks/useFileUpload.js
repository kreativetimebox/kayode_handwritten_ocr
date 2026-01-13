import { useState, useCallback } from 'react';
import uploadService from '../services/uploadService';

export const useFileUpload = () => {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFiles = useCallback(async (fileList) => {
    if (!fileList || fileList.length === 0) {
      setError('Please select files');
      return;
    }

    setFiles(Array.from(fileList));
    setError(null);
  }, []);

  const uploadFiles = useCallback(async (useMockData = false) => {
    if (!files || files.length === 0) {
      setError('No files selected');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      let result;

      if (useMockData) {
        // Simulate API delay
        await new Promise((resolve) => setTimeout(resolve, 1000));
        result = await uploadService.getMockData();
      } else {
        const formData = new FormData();
        files.forEach((file) => {
          formData.append('files', file);
        });
        result = await uploadService.uploadFiles(formData);
      }

      setUploadResult(result);
      setFiles([]);
      return result;
    } catch (err) {
      setError(err.message);
      console.error('Upload error:', err);
    } finally {
      setLoading(false);
    }
  }, [files]);

  const clearFiles = useCallback(() => {
    setFiles([]);
    setError(null);
  }, []);

  const clearResult = useCallback(() => {
    setUploadResult(null);
  }, []);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const { files: droppedFiles } = e.dataTransfer;
    handleFiles(droppedFiles);
  }, [handleFiles]);

  return {
    files,
    loading,
    error,
    uploadResult,
    dragActive,
    handleFiles,
    uploadFiles,
    clearFiles,
    clearResult,
    handleDrag,
    handleDrop,
  };
};