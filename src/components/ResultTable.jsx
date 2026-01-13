import React from "react";

const ResultTable = ({ data, onClose }) => {
  if (!data || !data.data || data.data.length === 0) {
    return null;
  }

  const jsonData = data.data;
  const columns = jsonData.length > 0 ? Object.keys(jsonData[0]) : [];

  // Download as CSV
  const downloadCSV = () => {
    try {
      // Create CSV header
      const headers = columns.map((col) => `"${col}"`).join(",");

      // Create CSV rows
      const rows = jsonData.map((row) =>
        columns
          .map((col) => {
            const value = row[col];
            // Escape quotes and wrap in quotes if contains comma
            const escaped = String(value).replace(/"/g, '""');
            return `"${escaped}"`;
          })
          .join(",")
      );

      const csv = [headers, ...rows].join("\n");
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const link = document.createElement("a");
      const url = URL.createObjectURL(blob);

      link.setAttribute("href", url);
      link.setAttribute("download", "upload-results.csv");
      link.style.visibility = "hidden";

      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error("CSV download error:", error);
      alert("Failed to download CSV");
    }
  };

  const downloadExcel = () => {
    try {
      let html = `
      <html xmlns:o="urn:schemas-microsoft-com:office:office"
            xmlns:x="urn:schemas-microsoft-com:office:excel"
            xmlns="http://www.w3.org/TR/REC-html40">
      <head>
        <meta charset="UTF-8">
      </head>
      <body>
        <table border="1">
          <tr>
    `;

      // Headers
      columns.forEach((col) => {
        html += `
        <th style="background:#22c55e;color:white;padding:8px;">
          ${col}
        </th>
      `;
      });

      html += `</tr>`;

      // Rows
      jsonData.forEach((row) => {
        html += `<tr>`;
        columns.forEach((col) => {
          const value = row[col] ?? ""; // handle null / undefined
          html += `
          <td style="padding:8px;">
            ${value === null || value === undefined ? '&nbsp;' : String(value)}
          </td>
        `;
        });
        html += `</tr>`;
      });

      html += `
        </table>
      </body>
      </html>
    `;

      const blob = new Blob([html], {
        type: "application/vnd.ms-excel",
      });

      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");

      link.href = url;
      link.download = "upload-results.xls"; // ✅ IMPORTANT
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Excel download error:", err);
      alert("Failed to download Excel");
    }
  };

  return (
    <div className="w-full mt-8">
      <div className="bg-white border border-green-200 rounded-lg overflow-hidden shadow-lg">
        {/* Header */}
        <div className="bg-green-600 px-6 py-4 flex justify-between items-center">
          <div>
            <h3 className="text-lg font-semibold text-white">Upload Results</h3>
            <p className="text-green-100 text-sm mt-1">
              {data.message} - Total Files: {data.totalFiles}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-white hover:bg-green-700 px-4 py-2 rounded-lg transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-green-50 border-b border-green-200">
                {columns.map((column) => (
                  <th
                    key={column}
                    className="px-6 py-3 text-left text-xs font-semibold text-gray-700 uppercase tracking-wider"
                  >
                    {column.replace(/([A-Z])/g, " $1").trim()}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-green-100">
              {jsonData.map((row, rowIndex) => (
                <tr
                  key={rowIndex}
                  className="hover:bg-green-50 transition-colors"
                >
                  {columns.map((column) => (
                    <td
                      key={`${rowIndex}-${column}`}
                      className="px-6 py-4 text-sm text-gray-700"
                    >
                      <div className="flex items-center space-x-2">
                        {column === "status" && (
                          <span
                            className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                              row[column] === "Success"
                                ? "bg-green-100 text-green-800"
                                : "bg-red-100 text-red-800"
                            }`}
                          >
                            {row[column]}
                          </span>
                        )}
                        {column !== "status" && (
                          <span>{String(row[column])}</span>
                        )}
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Footer */}
        <div className="bg-green-50 px-6 py-4 border-t border-green-200 flex justify-between items-center flex-wrap gap-4">
          <p className="text-sm text-gray-600">
            Showing {jsonData.length} of {data.totalFiles} files
          </p>
          <div className="flex gap-2 flex-wrap">
            <button
              onClick={() => {
                const jsonStr = JSON.stringify(data, null, 2);
                const element = document.createElement("a");
                element.setAttribute(
                  "href",
                  "data:application/json;charset=utf-8," +
                    encodeURIComponent(jsonStr)
                );
                element.setAttribute("download", "upload-results.json");
                element.style.display = "none";
                document.body.appendChild(element);
                element.click();
                document.body.removeChild(element);
              }}
              className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors text-sm font-medium flex items-center gap-2"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
              JSON
            </button>

            <button
              onClick={downloadCSV}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors text-sm font-medium flex items-center gap-2"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
              CSV
            </button>

            <button
              onClick={downloadExcel}
              className="px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors text-sm font-medium flex items-center gap-2"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
              Excel
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultTable;
