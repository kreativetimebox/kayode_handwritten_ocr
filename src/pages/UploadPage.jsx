import React from 'react';
import Layout from '../components/Layout';
import FileUploadZone from '../components/FileUploadZone';
import FileList from '../components/FileList';
import ResultTable from '../components/ResultTable';
import { useFileUpload } from '../hooks/useFileUpload';

const UploadPage = () => {
  const {
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
  } = useFileUpload();

  const handleUploadClick = async (useMock = false) => {
    await uploadFiles(useMock);
  };

  return (
    <Layout
      title="📁 File Upload Manager"
      subtitle="Upload single files, multiple files, or entire folders with drag and drop support"
    >
      {/* Error Alert */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-start">
            <div className="text-red-500 mr-3 mt-0.5">
              <svg
                className="w-5 h-5"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <div>
              <h3 className="font-semibold text-red-800">Error</h3>
              <p className="text-red-700 text-sm mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Success Alert */}
      {uploadResult && uploadResult.success && (
        <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-start">
            <div className="text-green-500 mr-3 mt-0.5">
              <svg
                className="w-5 h-5"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <div>
              <h3 className="font-semibold text-green-800">Success</h3>
              <p className="text-green-700 text-sm mt-1">
                {uploadResult.message}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Main Upload Section */}
      <div className="bg-white rounded-lg shadow-md p-8 mb-8">
        <FileUploadZone
          dragActive={dragActive}
          onDrag={handleDrag}
          onDrop={handleDrop}
          onFileSelect={handleFiles}
          disabled={loading}
        />

        {/* File List */}
        {files.length > 0 && <FileList files={files} onClear={clearFiles} />}

        {/* Action Buttons */}
        {files.length > 0 && (
          <div className="mt-6 flex gap-4 justify-center flex-wrap">
            <button
              onClick={() => handleUploadClick(false)}
              disabled={loading || files.length === 0}
              className="px-8 py-3 bg-green-500 text-white rounded-lg font-semibold hover:bg-green-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              {loading ? (
                <>
                  <div className="animate-spin">⟳</div>
                  Uploading...
                </>
              ) : (
                <>
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                    />
                  </svg>
                  Upload Files
                </>
              )}
            </button>

            <button
              onClick={() => handleUploadClick(true)}
              disabled={loading || files.length === 0}
              className="px-8 py-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              Test with Mock Data
            </button>

            <button
              onClick={clearFiles}
              disabled={loading || files.length === 0}
              className="px-8 py-3 bg-white text-gray-700 border-2 border-gray-300 rounded-lg font-semibold hover:bg-gray-50 disabled:bg-gray-100 disabled:cursor-not-allowed transition-colors"
            >
              Clear
            </button>
          </div>
        )}
      </div>

      {/* Result Table */}
      {uploadResult && (
        <ResultTable data={uploadResult} onClose={clearResult} />
      )}

      {/* Info Box */}
      {files.length === 0 && !uploadResult && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-green-900 mb-2">
            How to use:
          </h3>
          <ul className="text-green-800 space-y-2 list-disc list-inside">
            <li>Drag and drop files or folders directly onto the upload zone</li>
            <li>Click "Choose Files" to select individual files</li>
            <li>Click "Choose Folder" to upload an entire folder</li>
            <li>
              Click "Upload Files" to upload to the server or "Test with Mock
              Data" for testing
            </li>
            <li>View results in the table and download as JSON</li>
          </ul>
        </div>
      )}
    </Layout>
  );
};

export default UploadPage;