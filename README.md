### Installation with Default Configuration
- **If Ollama is on a Different Server**, use this command:

  To connect to Ollama on another server, change the `OLLAMA_BASE_URL` to the server's URL:

  ```bash
  docker run -d -p 3000:8080 -e OLLAMA_BASE_URL=https://sandbox.lethanhhung.site -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
  ```
  
không rõ có cần  
USE_OLLAMA  
để cài Ollama cho container hay không (đang chỉ có webui thôi) ?  
