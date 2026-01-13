import React, { useRef } from 'react';

const FileUploadZone = ({
  dragActive,
  onDrag,
  onDrop,
  onFileSelect,
  disabled = false,
}) => {
  const fileInputRef = useRef(null);
  const folderInputRef = useRef(null);

  const handleFileClick = () => {
    fileInputRef.current?.click();
  };

  const handleFolderClick = () => {
    folderInputRef.current?.click();
  };

  const handleFileInputChange = (e) => {
    onFileSelect(e.target.files);
    e.target.value = ''; // Reset input
  };

  const handleFolderInputChange = (e) => {
    onFileSelect(e.target.files);
    e.target.value = ''; // Reset input
  };

  return (
    <div className="w-full">
      <div
        className={`relative border-2 border-dashed rounded-lg p-8 transition-all duration-300 ${
          dragActive
            ? 'border-green-500 bg-green-50'
            : 'border-green-300 bg-white hover:border-green-400'
        } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
        onDragEnter={onDrag}
        onDragLeave={onDrag}
        onDragOver={onDrag}
        onDrop={onDrop}
      >
        <div className="flex flex-col items-center justify-center space-y-4">
          {/* Icon */}
          <div className="text-green-500">
            <svg
              className="w-16 h-16"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M12 4v16m8-8H4"
              />
            </svg>
          </div>

          {/* Text Content */}
          <div className="text-center">
            <p className="text-lg font-semibold text-gray-800">
              {dragActive ? 'Drop files here' : 'Drag and drop files here'}
            </p>
            <p className="text-sm text-gray-600 mt-1">
              or use the buttons below to select files
            </p>
          </div>

          {/* Buttons */}
          <div className="flex flex-col sm:flex-row gap-3 pt-4">
            <button
              onClick={handleFileClick}
              disabled={disabled}
              className="px-6 py-2 bg-green-500 text-white rounded-lg font-medium hover:bg-green-600 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              Choose Files
            </button>

            <button
              onClick={handleFolderClick}
              disabled={disabled}
              className="px-6 py-2 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              Choose Folder
            </button>
          </div>

          {/* Hidden Inputs */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleFileInputChange}
            className="hidden"
            disabled={disabled}
          />

          <input
            ref={folderInputRef}
            type="file"
            webkitdirectory="true"
            mozdirectory="true"
            onChange={handleFolderInputChange}
            className="hidden"
            disabled={disabled}
          />
        </div>
      </div>
    </div>
  );
};

export default FileUploadZone;