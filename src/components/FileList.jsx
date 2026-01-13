import React from 'react';

const FileList = ({ files, onClear }) => {
  if (!files || files.length === 0) {
    return null;
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const getFileIcon = (fileName) => {
    const ext = fileName.split('.').pop().toLowerCase();
    const iconMap = {
      pdf: '📄',
      doc: '📝',
      docx: '📝',
      xlsx: '📊',
      xls: '📊',
      jpg: '🖼️',
      jpeg: '🖼️',
      png: '🖼️',
      gif: '🖼️',
      zip: '🗜️',
      rar: '🗜️',
      txt: '📃',
    };
    return iconMap[ext] || '📦';
  };

  return (
    <div className="w-full mt-6">
      <div className="bg-white border border-green-200 rounded-lg overflow-hidden">
        {/* Header */}
        <div className="bg-green-50 px-6 py-4 border-b border-green-200">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold text-gray-800">
              Selected Files ({files.length})
            </h3>
            <button
              onClick={onClear}
              className="text-sm px-4 py-2 text-red-500 hover:bg-red-50 rounded-lg transition-colors"
            >
              Clear All
            </button>
          </div>
        </div>

        {/* File List */}
        <div className="divide-y divide-green-100 max-h-96 overflow-y-auto">
          {files.map((file, index) => (
            <div
              key={index}
              className="px-6 py-4 hover:bg-green-50 transition-colors flex items-center justify-between"
            >
              <div className="flex items-center space-x-4 flex-1 min-w-0">
                <span className="text-2xl flex-shrink-0">
                  {getFileIcon(file.name)}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium text-gray-800 truncate">
                    {file.name}
                  </p>
                  <p className="text-xs text-gray-500">
                    {formatFileSize(file.size)}
                  </p>
                </div>
              </div>
              <div className="flex-shrink-0 ml-4">
                <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                  Ready
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default FileList;