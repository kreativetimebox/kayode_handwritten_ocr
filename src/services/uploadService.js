import axios from "axios";

const API_BASE_URL = "http://localhost:5000/api";

const uploadService = {
  // Single endpoint that handles all file uploads
  uploadFiles: async (formData) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          return percentCompleted;
        },
      });
      return response.data;
    } catch (error) {
      throw new Error(
        error.response?.data?.message || "Failed to upload files"
      );
    }
  },

  // Get mock data for testing (dummy response)
  getMockData: async () => {
    return {
      success: true,
      message: "Mock data loaded",
      data: [
        {
          file_name: "DDR Form - 6B-20250908-compressed.pdf",
          "invoice(Y/N)": "N",
          "Visit(Y/N)": "N",
          Transit_No: 6122,
          Account_No: "09-67106",
          prefix: "6B",
          ILR_date: 20250908,
          amount: 407.0,
          other_amount: 46.0,
        },
        {
          file_name: "DDR Form - W0-20250805.pdf",
          "invoice(Y/N)": "Y",
          "Visit(Y/N)": "N",
          Transit_No: 342,
          Account_No: "09-66509",
          prefix: null,
          ILR_date: null,
          amount: null,
          other_amount: null,
        },
      ],
      totalFiles: 2,
    };
  },
};

export default uploadService;
