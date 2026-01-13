/**
 * Export Utilities for different file formats
 * Handles JSON, CSV, and Excel exports
 */

/**
 * Download data as JSON
 * @param {Object} data - Data to export
 * @param {String} filename - Filename for download
 */
export const downloadJSON = (data, filename = 'data.json') => {
  try {
    const jsonStr = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json;charset=utf-8;' });
    createDownloadLink(blob, filename);
  } catch (error) {
    console.error('JSON export error:', error);
    throw new Error('Failed to export JSON');
  }
};

/**
 * Download data as CSV
 * @param {Array} data - Array of objects to export
 * @param {String} filename - Filename for download
 */
export const downloadCSV = (data, filename = 'data.csv') => {
  try {
    if (!Array.isArray(data) || data.length === 0) {
      throw new Error('Data must be a non-empty array');
    }

    const columns = Object.keys(data[0]);
    
    // Create CSV header
    const headers = columns.map(col => `"${col}"`).join(',');
    
    // Create CSV rows
    const rows = data.map(row => 
      columns.map(col => {
        const value = row[col];
        // Escape quotes and wrap in quotes
        const escaped = String(value).replace(/"/g, '""');
        return `"${escaped}"`;
      }).join(',')
    );

    const csv = [headers, ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    createDownloadLink(blob, filename);
  } catch (error) {
    console.error('CSV export error:', error);
    throw new Error('Failed to export CSV');
  }
};

/**
 * Download data as Excel (HTML format for compatibility)
 * @param {Array} data - Array of objects to export
 * @param {String} filename - Filename for download
 * @param {String} sheetName - Name of the Excel sheet
 */
export const downloadExcel = (data, filename = 'data.xlsx', sheetName = 'Sheet1') => {
  try {
    if (!Array.isArray(data) || data.length === 0) {
      throw new Error('Data must be a non-empty array');
    }

    const columns = Object.keys(data[0]);
    
    // Create Excel HTML format
    let html = `<html>
      <head>
        <meta charset="UTF-8">
        <style>
          body { font-family: Arial, sans-serif; }
          table { border-collapse: collapse; width: 100%; }
          th {
            background-color: #22c55e;
            color: white;
            font-weight: bold;
            padding: 10px;
            border: 1px solid #ddd;
            text-align: left;
          }
          td {
            padding: 8px;
            border: 1px solid #ddd;
          }
          tr:nth-child(even) {
            background-color: #f0fdf4;
          }
          tr:hover {
            background-color: #dcfce7;
          }
        </style>
      </head>
      <body>
        <table>
          <thead>
            <tr>`;
    
    // Add headers
    columns.forEach(col => {
      html += `<th>${escapeHtml(col)}</th>`;
    });
    html += '</tr></thead><tbody>';

    // Add data rows
    data.forEach(row => {
      html += '<tr>';
      columns.forEach(col => {
        html += `<td>${escapeHtml(String(row[col]))}</td>`;
      });
      html += '</tr>';
    });

    html += `</tbody></table></body></html>`;

    const blob = new Blob([html], { type: 'application/vnd.ms-excel;charset=utf-8;' });
    createDownloadLink(blob, filename);
  } catch (error) {
    console.error('Excel export error:', error);
    throw new Error('Failed to export Excel');
  }
};

/**
 * Create and trigger download link
 * @param {Blob} blob - File blob
 * @param {String} filename - Filename for download
 */
const createDownloadLink = (blob, filename) => {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  
  document.body.appendChild(link);
  link.click();
  
  // Cleanup
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

/**
 * Escape HTML special characters
 * @param {String} text - Text to escape
 * @returns {String} Escaped text
 */
const escapeHtml = (text) => {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, m => map[m]);
};

/**
 * Export data in multiple formats
 * @param {Object} data - Data object with 'data' array property
 * @param {String} baseFilename - Base filename without extension
 */
export const exportAll = (data, baseFilename = 'upload-results') => {
  const timestamp = new Date().toISOString().slice(0, 10);
  const filename = `${baseFilename}-${timestamp}`;

  try {
    downloadJSON(data, `${filename}.json`);
    downloadCSV(data.data, `${filename}.csv`);
    downloadExcel(data.data, `${filename}.xlsx`);
  } catch (error) {
    console.error('Export all error:', error);
    throw new Error('Failed to export all formats');
  }
};