// import Layout from "../layout/Layout.jsx";
// import HeaderSearch from "../components/HeaderSearch.jsx";

// import ProfileCard from "../features/sidebar/ProfileCard.jsx";
// import MentorsCard from "../features/sidebar/MentorsCard.jsx";
// import CommunitiesCard from "../features/sidebar/CommunitiesCard.jsx";

// import Tabs from "../features/social/components/Tabs.jsx";
// import FeedList from "../features/feed/FeedList.jsx";
// import {Routes, Route} from "react-router-dom"
import {Routes, Route } from "react-router-dom"
import Home from "../pages/Home.jsx"
import ProfilePage from "../pages/ProfilePage.jsx"

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />}/>
      <Route path="/profile" element={<ProfilePage />} />
    </Routes>
  );
  // return <div style={{ padding: 16 }}>Hello world</div>;
}