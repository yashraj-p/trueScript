import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Body from './components/Body'
function App() {
  const [count, setCount] = useState(0)

  return (
    <div >
      <Navbar/>
      <Body/>
      <Footer/>
    </div>
  )
}

export default App
