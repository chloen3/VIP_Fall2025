import Layout from "../layout/Layout.jsx";
import HeaderSearch from "../components/HeaderSearch.jsx";

import ProfileCard from "../features/sidebar/ProfileCard.jsx";
import MentorsCard from "../features/sidebar/MentorsCard.jsx";
import CommunitiesCard from "../features/sidebar/CommunitiesCard.jsx";
import SocialPanel from "../features/social/SocialPanel.jsx";

import Tabs from "../features/social/components/Tabs.jsx";
import FeedList from "../features/social/components/FeedList.jsx";

export default function Home() {
    return (
          <Layout
            header={<HeaderSearch />}
            sidebar={
              <>
                <ProfileCard />
                <MentorsCard />
                <CommunitiesCard />
              </>
            }
          >
            <SocialPanel />
          </Layout>
      );
}