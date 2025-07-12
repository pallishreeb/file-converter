# Use Node.js base image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy the entire project
COPY . .

# Build TypeScript code
RUN npm run build

# Expose the port
EXPOSE 5000

# Start the server
CMD ["node", "dist/index.js"]
