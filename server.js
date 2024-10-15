const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = socketIo(server);

app.use(express.json());

// Serve static files from the 'client' directory
app.use('/client', express.static(path.join(__dirname, 'client')));

// Serve client.html at the '/client' route
app.get('/client', (req, res) => {
    res.sendFile(path.join(__dirname, 'client', 'client.html'));
});

// Serve static files from the 'server' directory
app.use('/server', express.static(path.join(__dirname, 'server')));

// Serve server.html at the '/server' route
app.get('/server', (req, res) => {
    res.sendFile(path.join(__dirname, 'server', 'server.html'));
});

// Endpoint to receive location data
app.post('/update-location', (req, res) => {
    const location = req.body;
    console.log('Received location:', location);

    // Broadcast location to all connected clients
    io.emit('update-location', location);

    res.json({ status: 'success' });
});

io.on('connection', (socket) => {
    console.log('New client connected');
    socket.on('disconnect', () => {
        console.log('Client disconnected');
    });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
