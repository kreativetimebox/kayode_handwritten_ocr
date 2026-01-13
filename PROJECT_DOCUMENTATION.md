# 📚 Complete Project Documentation - File Upload UI

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [File Structure](#file-structure)
5. [Component Documentation](#component-documentation)
6. [Services Documentation](#services-documentation)
7. [Hooks Documentation](#hooks-documentation)
8. [Backend Documentation](#backend-documentation)
9. [API Reference](#api-reference)
10. [Data Flow](#data-flow)
11. [Styling & Theme](#styling--theme)
12. [Error Handling](#error-handling)
13. [Performance Optimization](#performance-optimization)
14. [Deployment Guide](#deployment-guide)
15. [FAQ & Troubleshooting](#faq--troubleshooting)

---

## Project Overview

### What is This Project?
A modern, modular React application for uploading single files, multiple files, or entire folders with drag-and-drop functionality. The application accepts JSON responses from a backend and displays them in a formatted table view.

### Key Features
- 🎯 **Drag & Drop Upload** - Intuitive file handling
- 📁 **Multi-file Upload** - Upload multiple files at once
- 📂 **Folder Upload** - Upload entire directory structures
- 📊 **JSON to Table** - Convert API responses to tables
- 🎨 **Green & White Theme** - Professional color scheme
- 📱 **Responsive Design** - Mobile, tablet, desktop support
- ⚡ **Mock Data Testing** - Test without backend
- 💾 **JSON Export** - Download results
- 🔄 **Real-time Updates** - Instant UI feedback

### Technology Stack
```
Frontend:
  - React 18.x (UI Framework)
  - Tailwind CSS 3.x (Styling)
  - Axios (HTTP Client)
  - Hooks (State Management)

Backend:
  - Node.js (Runtime)
  - Express 4.x (Web Framework)
  - Multer 1.x (File Upload)
  - CORS (Cross-Origin)
```

---

## Architecture

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    BROWSER (Frontend)                    │
├─────────────────────────────────────────────────────────┤
│  React Components (FileUploadZone, FileList, Table)    │
│              ↓                                           │
│  Custom Hooks (useFileUpload)                          │
│              ↓                                           │
│  Services (uploadService)                              │
│              ↓                                           │
│  Axios HTTP Client                                     │
└─────────────────────────────────────────────────────────┘
                        ↓ HTTP
┌─────────────────────────────────────────────────────────┐
│              SERVER (Backend - Port 5000)               │
├─────────────────────────────────────────────────────────┤
│  Express Server                                         │
│       ↓                                                  │
│  POST /api/upload Route                               │
│       ↓                                                  │
│  Multer File Handler                                   │
│       ↓                                                  │
│  File Storage (uploads/ folder)                       │
│       ↓                                                  │
│  JSON Response                                         │
└─────────────────────────────────────────────────────────┘
```

### Component Hierarchy
```
App.jsx
└── UploadPage.jsx
    ├── Layout.jsx
    │   ├── Header (Title & Subtitle)
    │   ├── Main Content (children)
    │   └── Footer
    ├── FileUploadZone.jsx (Drag & Drop)
    ├── FileList.jsx (Selected Files)
    └── ResultTable.jsx (Upload Results)
```

---

## Installation & Setup

### Prerequisites
- Node.js 14+ installed
- npm or yarn package manager
- Code editor (VS Code recommended)
- Git (optional)

### Step-by-Step Installation

#### Phase 1: Frontend Setup

**Step 1: Create React Project**
```bash
npx create-react-app file-upload-ui
cd file-upload-ui
```

**Step 2: Install Dependencies**
```bash
npm install axios tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

**Step 3: Configure Tailwind CSS**

Edit `tailwind.config.js`:
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        green: {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#145231',
        }
      }
    },
  },
  plugins: [],
}
```

Edit `src/index.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans',
    'Helvetica Neue', sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
```

**Step 4: Create Folder Structure**
```bash
mkdir src/components
mkdir src/pages
mkdir src/services
mkdir src/hooks
```

**Step 5: Copy Component Files**
Create all files according to the File Structure section below.

**Step 6: Start Frontend**
```bash
npm start
```
Frontend will run on `http://localhost:3000`

#### Phase 2: Backend Setup (Optional)

**Step 1: Create Backend Directory**
```bash
mkdir backend
cd backend
npm init -y
```

**Step 2: Install Backend Dependencies**
```bash
npm install express cors multer
```

**Step 3: Create Backend Files**
Create `server.js` and `routes/upload.js` according to Backend Documentation section.

**Step 4: Create uploads Directory**
```bash
mkdir uploads
```

**Step 5: Start Backend**
```bash
node server.js
```
Backend will run on `http://localhost:5000`

---

## File Structure

### Complete Directory Tree
```
file-upload-ui/
│
├── node_modules/                    # Dependencies
│
├── public/
│   ├── index.html                   # Main HTML
│   └── favicon.ico
│
├── src/
│   │
│   ├── components/                  # Reusable Components
│   │   ├── FileUploadZone.jsx      # Drag & drop zone
│   │   ├── FileList.jsx             # List of selected files
│   │   ├── ResultTable.jsx          # JSON response table
│   │   └── Layout.jsx               # Main layout wrapper
│   │
│   ├── pages/                       # Page Components
│   │   └── UploadPage.jsx           # Main page
│   │
│   ├── services/                    # API Services
│   │   └── uploadService.js         # Axios API calls
│   │
│   ├── hooks/                       # Custom Hooks
│   │   └── useFileUpload.js         # File upload logic
│   │
│   ├── App.jsx                      # App component
│   ├── App.css
│   ├── index.css                    # Tailwind CSS
│   └── index.js                     # React entry point
│
├── .gitignore
├── package.json                     # NPM dependencies
├── package-lock.json
├── tailwind.config.js               # Tailwind config
└── postcss.config.js                # PostCSS config

backend/                            # Backend Folder (separate)
│
├── routes/
│   └── upload.js                   # Upload route
│
├── server.js                       # Express server
├── package.json
├── package-lock.json
└── uploads/                        # Uploaded files storage
    └── (files stored here)
```

---

## Component Documentation

### 1. Layout.jsx
**Purpose:** Main layout wrapper with header and footer

**Props:**
```javascript
{
  children: React.ReactNode,        // Main content
  title: string,                     // Page title
  subtitle: string                   // Optional subtitle
}
```

**Usage:**
```jsx
<Layout 
  title="File Upload Manager" 
  subtitle="Upload files with drag and drop"
>
  {/* Children content */}
</Layout>
```

**Features:**
- Sticky header with gradient background
- Responsive container (max-width: 6xl)
- Footer with copyright info
- Gradient background

---

### 2. FileUploadZone.jsx
**Purpose:** Main drag-and-drop upload area

**Props:**
```javascript
{
  dragActive: boolean,               // Is dragging?
  onDrag: function,                  // Handle drag events
  onDrop: function,                  // Handle drop event
  onFileSelect: function,            // Handle file selection
  disabled: boolean                  // Disable interaction
}
```

**Usage:**
```jsx
<FileUploadZone
  dragActive={dragActive}
  onDrag={handleDrag}
  onDrop={handleDrop}
  onFileSelect={handleFiles}
  disabled={loading}
/>
```

**Features:**
- Drag-and-drop zone with visual feedback
- "Choose Files" button (single/multiple)
- "Choose Folder" button (webkitdirectory support)
- Responsive layout
- Hidden file inputs
- Disabled state styling

---

### 3. FileList.jsx
**Purpose:** Display selected files before upload

**Props:**
```javascript
{
  files: File[],                     // Array of File objects
  onClear: function                  // Clear button callback
}
```

**Usage:**
```jsx
<FileList 
  files={files} 
  onClear={clearFiles}
/>
```

**Features:**
- File icons based on extension
- File size formatting (Bytes, KB, MB, GB)
- "Ready" status badge
- Max height with scroll
- File count display
- Clear All button
- Hover effects

---

### 4. ResultTable.jsx
**Purpose:** Display upload results as table

**Props:**
```javascript
{
  data: {                            // Upload response
    success: boolean,
    message: string,
    data: Array<Object>,
    totalFiles: number
  },
  onClose: function                  // Close button callback
}
```

**Usage:**
```jsx
<ResultTable 
  data={uploadResult} 
  onClose={clearResult}
/>
```

**Features:**
- Dynamic column generation from JSON keys
- Auto-formatting column headers
- Status badge styling (Success/Failed)
- Download JSON button
- Scrollable table for mobile
- Result summary footer
- Close button

**JSON Format Expected:**
```json
{
  "success": true,
  "message": "Files uploaded successfully",
  "data": [
    {
      "id": 1,
      "fileName": "document.pdf",
      "size": "2048.50 KB",
      "uploadDate": "01/13/2026",
      "status": "Success"
    }
  ],
  "totalFiles": 1
}
```

---

### 5. UploadPage.jsx
**Purpose:** Main page orchestrating all components

**State & Logic:**
- File selection and management
- Upload execution
- Error handling
- Result display

**Features:**
- Integration of all components
- Alert displays (error/success)
- Upload and mock data buttons
- Info box for guidance
- Loading states
- Error messages

---

## Services Documentation

### uploadService.js
**Purpose:** Centralized API communication using Axios

**Config:**
```javascript
const API_BASE_URL = 'http://localhost:5000/api';
```

**Methods:**

#### 1. uploadService.uploadFiles(formData)
**Purpose:** Upload files to backend

**Parameters:**
```javascript
formData: FormData {
  files: File[]  // Multiple files
}
```

**Returns:**
```javascript
Promise<{
  success: boolean,
  message: string,
  data: Array<Object>,
  totalFiles: number
}>
```

**Example:**
```javascript
const formData = new FormData();
files.forEach(file => {
  formData.append('files', file);
});

const result = await uploadService.uploadFiles(formData);
```

#### 2. uploadService.getMockData()
**Purpose:** Return dummy data for testing

**Returns:**
```javascript
Promise<{
  success: true,
  message: "Mock data loaded",
  data: [
    {
      id: 1,
      fileName: "document.pdf",
      uploadedName: "document-1705088400000.pdf",
      size: "2048.50 KB",
      uploadDate: "01/13/2026",
      status: "Success",
      mimeType: "application/pdf"
    },
    // ... more items
  ],
  totalFiles: 3
}>
```

**Example:**
```javascript
const mockResult = await uploadService.getMockData();
```

**Error Handling:**
```javascript
try {
  const result = await uploadService.uploadFiles(formData);
} catch (error) {
  console.error(error.message);
  // "Failed to upload files" or custom message
}
```

---

## Hooks Documentation

### useFileUpload()
**Purpose:** Custom hook managing file upload logic

**Returns:**
```javascript
{
  // State
  files: File[],                     // Selected files
  loading: boolean,                  // Upload in progress
  error: string | null,              // Error message
  uploadResult: Object | null,       // Upload response
  dragActive: boolean,               // Is dragging?

  // Methods
  handleFiles: (FileList) => void,  // Process selected files
  uploadFiles: (useMockData) => Promise<Object>,
  clearFiles: () => void,            // Clear file selection
  clearResult: () => void,           // Clear upload result
  handleDrag: (DragEvent) => void,  // Handle drag events
  handleDrop: (DragEvent) => void   // Handle drop event
}
```

**Usage:**
```javascript
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
```

**State Variables:**

| Variable | Type | Description |
|----------|------|-------------|
| `files` | `File[]` | Array of selected File objects |
| `loading` | `boolean` | True while uploading |
| `error` | `string \| null` | Error message if upload fails |
| `uploadResult` | `Object \| null` | Response from API |
| `dragActive` | `boolean` | True when dragging over zone |

**Methods:**

| Method | Purpose |
|--------|---------|
| `handleFiles(fileList)` | Process files from input/drop |
| `uploadFiles(useMock)` | Send files to server |
| `clearFiles()` | Reset file selection |
| `clearResult()` | Reset upload result |
| `handleDrag(event)` | Handle drag enter/leave |
| `handleDrop(event)` | Handle drop event |

---

## Backend Documentation

### Server Setup

**File:** `backend/server.js`

```javascript
const express = require('express');
const cors = require('cors');
const uploadRoutes = require('./routes/upload');

const app = express();
const PORT = 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api', uploadRoutes);

// Health check
app.get('/', (req, res) => {
  res.json({ message: 'Server is running' });
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
```

**Environment Variables:**
```
PORT=5000
```

### Upload Route

**File:** `backend/routes/upload.js`

**Multer Configuration:**
```javascript
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, '../uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const timestamp = Date.now();
    const ext = path.extname(file.originalname);
    const name = path.basename(file.originalname, ext);
    cb(null, `${name}-${timestamp}${ext}`);
  }
});

const upload = multer({ 
  storage,
  limits: { fileSize: 50 * 1024 * 1024 } // 50MB
});
```

**File Naming Strategy:**
- Pattern: `{originalName}-{timestamp}{ext}`
- Example: `document-1705088400000.pdf`
- Prevents filename collisions

**Limitations:**
- Max file size: 50MB
- Supported formats: All formats
- Max files per request: Unlimited

---

## API Reference

### Upload Endpoint

**Endpoint:** `POST /api/upload`

**Base URL:** `http://localhost:5000`

**Content-Type:** `multipart/form-data`

**Request:**
```javascript
const formData = new FormData();
formData.append('files', file1);
formData.append('files', file2);
formData.append('files', file3);

POST /api/upload
Content-Type: multipart/form-data
Body: formData
```

**cURL Example:**
```bash
curl -X POST \
  -F "files=@document.pdf" \
  -F "files=@image.jpg" \
  http://localhost:5000/api/upload
```

**Response (Success - 200):**
```json
{
  "success": true,
  "message": "Files uploaded successfully",
  "data": [
    {
      "id": 1,
      "fileName": "document.pdf",
      "uploadedName": "document-1705088400000.pdf",
      "size": "2048.50 KB",
      "uploadDate": "01/13/2026",
      "status": "Success",
      "mimeType": "application/pdf"
    },
    {
      "id": 2,
      "fileName": "image.jpg",
      "uploadedName": "image-1705088401000.jpg",
      "size": "1024.25 KB",
      "uploadDate": "01/13/2026",
      "status": "Success",
      "mimeType": "image/jpeg"
    }
  ],
  "totalFiles": 2
}
```

**Response (Error - 400):**
```json
{
  "success": false,
  "message": "No files uploaded"
}
```

**Response (Error - 500):**
```json
{
  "success": false,
  "message": "Server error during file upload"
}
```

---

## Data Flow

### Complete Upload Flow

```
1. USER INTERACTION
   ├── Drag files over zone
   │   └── dragActive = true
   │       └── Zone highlights green
   │
   ├── Drop files
   │   └── handleDrop triggered
   │
   ├── Or Click "Choose Files"
   │   └── File input opens
   │
   ├── Select files
   │   └── handleFiles called
   │       └── files state updated
   │           └── FileList renders

2. FILE SELECTION
   ├── files[] populated
   ├── FileList component renders
   ├── Upload buttons enabled
   └── User can review files

3. UPLOAD INITIATION
   ├── User clicks "Upload Files"
   │   or "Test with Mock Data"
   ├── uploadFiles() called with useMockData flag
   ├── loading = true
   └── Buttons disabled

4. REQUEST PREPARATION
   ├── FormData created
   ├── Files appended
   └── Ready for transmission

5. HTTP REQUEST
   ├── Axios POST to /api/upload
   ├── Content-Type: multipart/form-data
   ├── Upload progress tracked
   └── Request sent to server

6. SERVER PROCESSING
   ├── Express receives POST
   ├── Multer processes files
   ├── Files stored in uploads/
   ├── Response data generated
   └── JSON returned

7. RESPONSE HANDLING
   ├── uploadService receives response
   ├── uploadResult state updated
   ├── loading = false
   ├── error cleared
   └── files cleared

8. UI UPDATE
   ├── ResultTable renders
   ├── Success alert shown
   ├── JSON data displayed as table
   └── Download JSON option available
```

### State Management Flow
```
User Input
    ↓
Hook Handler (handleFiles, uploadFiles)
    ↓
State Update (files, loading, error, uploadResult)
    ↓
Component Re-render
    ↓
UI Display
```

---

## Styling & Theme

### Color Palette
```
Primary Green: #22c55e (500)
Dark Green: #16a34a (600)
Darker Green: #15803d (700)
Light Green: #bbf7d0 (200)
Lighter Green: #dcfce7 (50)

White: #ffffff
Gray: #6b7280 (600)
```

### Tailwind Classes Used
```
Spacing: p-4, py-6, px-8, gap-2
Typography: text-lg, font-semibold, text-green-700
Layouts: flex, grid, max-w-6xl
Effects: rounded-lg, shadow-md, transition-colors
States: hover:, disabled:, focus:
Responsive: sm:, md:, lg:
```

### Theme Customization
Edit `tailwind.config.js`:
```javascript
theme: {
  extend: {
    colors: {
      primary: '#22c55e', // Change main color
      secondary: '#16a34a'
    },
    spacing: {
      // Custom spacing
    }
  }
}
```

---

## Error Handling

### Frontend Error Handling

**Try-Catch Blocks:**
```javascript
try {
  const result = await uploadService.uploadFiles(formData);
  setUploadResult(result);
} catch (err) {
  setError(err.message);
}
```

**Error Messages Displayed:**
- "Please select files"
- "No files selected"
- "Failed to upload files"
- Custom API error messages

**UI Error Display:**
```jsx
{error && (
  <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
    <h3 className="font-semibold text-red-800">Error</h3>
    <p className="text-red-700 text-sm mt-1">{error}</p>
  </div>
)}
```

### Backend Error Handling

**HTTP Status Codes:**
- 200: Success
- 400: No files uploaded
- 413: File too large
- 500: Server error

**Error Responses:**
```json
{
  "success": false,
  "message": "No files uploaded"
}
```

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| CORS Error | Backend not running | Start backend on port 5000 |
| File too large | > 50MB | Increase limit in multer config |
| Upload fails silently | Network error | Check browser console logs |
| Files not displaying | Wrong JSON format | Check API response structure |

---

## Performance Optimization

### Frontend Optimization

**1. Code Splitting**
```javascript
const UploadPage = React.lazy(() => import('./pages/UploadPage'));
```

**2. Memoization**
```javascript
const handleFiles = useCallback((fileList) => {
  // Process files
}, []);
```

**3. Conditional Rendering**
```javascript
{files.length > 0 && <FileList files={files} />}
```

**4. Event Debouncing**
Already optimized in drag handlers.

### Backend Optimization

**1. File Validation**
```javascript
limits: { fileSize: 50 * 1024 * 1024 }
```

**2. Async Processing**
```javascript
upload.array('files') // Async processing
```

**3. Efficient Naming**
```javascript
const timestamp = Date.now(); // Quick filename generation
```

### Network Optimization

**1. Upload Progress**
```javascript
onUploadProgress: (progressEvent) => {
  const percent = (progressEvent.loaded * 100) / progressEvent.total;
}
```

**2. Minimal Payload**
- Only necessary data sent
- Efficient JSON structure

**3. Error Recovery**
- Retry on network failure
- User-friendly error messages

---

## Deployment Guide

### Frontend Deployment (Netlify/Vercel)

**Step 1: Build Production Bundle**
```bash
npm run build
```

**Step 2: Deploy to Netlify**
```bash
npm install -g netlify-cli
netlify deploy --prod --dir=build
```

**Step 3: Configure Environment**
Create `.env.production`:
```
REACT_APP_API_URL=https://your-api.com/api
```

**Step 4: Update Service**
```javascript
const API_BASE_URL = process.env.REACT_APP_API_URL;
```

### Backend Deployment (Heroku/Railway)

**Step 1: Add Procfile**
```
web: node server.js
```

**Step 2: Update Port**
```javascript
const PORT = process.env.PORT || 5000;
```

**Step 3: Deploy**
```bash
git push heroku main
```

**Step 4: Set Environment**
```bash
heroku config:set NODE_ENV=production
```

### Docker Deployment

**Frontend Dockerfile:**
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

**Backend Dockerfile:**
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 5000
CMD ["node", "server.js"]
```

---

## FAQ & Troubleshooting

### Installation Issues

**Q: npm command not found**
```
A: Install Node.js from nodejs.org
```

**Q: Create React App fails**
```
A: Try: npx clear-npx-cache && npx create-react-app file-upload-ui
```

**Q: Tailwind not working**
```
A: Ensure tailwind.config.js exists and npm run start is used
```

### Runtime Issues

**Q: Port 3000 already in use**
```bash
# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Mac/Linux
lsof -i :3000
kill -9 <PID>
```

**Q: Port 5000 already in use**
```bash
# Same as above, change 3000 to 5000
```

**Q: CORS errors**
```
A: Backend must have CORS enabled:
   app.use(cors());
```

**Q: Files not uploading**
```
A: Check:
   1. Backend is running
   2. API URL is correct
   3. File size < 50MB
   4. Browser console for errors
```

### Performance Issues

**Q: Upload is slow**
```
A: 1. Check file sizes
   2. Check network speed
   3. Increase timeout in axios
```

**Q: UI is laggy**
```
A: 1. Enable React DevTools Profiler
   2. Check for unnecessary re-renders
   3. Use React.memo for components
```

### Data Issues

**Q: JSON table not displaying**
```
A: Check API response format matches expected structure
```

**Q: Downloaded JSON is empty**
```
A: Ensure uploadResult has valid data before download
```

### Production Issues

**Q: File upload fails in production**
```
A: 1. Check CORS headers
   2. Verify API URL in environment variables
   3. Check file size limits
```

---

## Best Practices

### Code Organization
- Keep components small and focused
- Use custom hooks for reusable logic
- Separate services from components
- Use layout components for consistency

### Error Handling
- Always wrap async operations in try-catch
- Display user-friendly error messages
- Log errors for debugging
- Validate input data

### Performance
- Use React.memo for unchanged components
- Implement useCallback for event handlers
- Lazy load heavy components
- Optimize bundle size

### Security
- Validate file types on client
- Validate files on server
- Set reasonable file size limits
- Use HTTPS in production

### Testing
- Unit test custom hooks
- Component tests for UI
- Integration tests for flows
- E2E tests for critical paths

---

## Future Enhancements

- [ ] Drag to reorder files
- [ ] File preview before upload
- [ ] Pause/resume uploads
- [ ] Batch operations
- [ ] File compression
- [ ] Progress percentage display
- [ ] Upload history
- [ ] File storage in database
- [ ] User authentication
- [ ] Role-based permissions
- [ ] Email notifications
- [ ] Advanced search/filter

---

## Support & Contact

For issues or questions:
1. Check FAQ section
2. Review error messages
3. Check browser console
4. Review code comments
5. Open GitHub issue

---

## License

MIT License - Free to use and modify

---

## Changelog

### Version 1.0.0 (Initial Release)
- Drag and drop functionality
- Multiple file upload
- Folder upload support
- JSON to table conversion
- Mock data testing
- Green & white theme
- Responsive design

---

*Last Updated: January 13, 2026*
*Documentation Version: 1.0.0*