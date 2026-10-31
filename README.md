# DTLogExtSim

## How To Run
1. **Install Docker:**  
    Make sure Docker is installed on your system. You can download it from [https://www.docker.com](https://www.docker.com).
2. **Clone or Download the Project**
3. **Run the Service:**  
    In the directory containing `docker-compose.yml`, run:
    ```bash
    docker compose up --build
    ```
    This command ensures you’re running the latest version of your app by rebuilding the Docker images before launching the containers. It’s useful when you’ve made changes to the source code or Dockerfile and want to see the updated behavior immediately.
4. **Access the UI:**
    Open your browser and navigate to:
    http://127.0.0.1:6660
