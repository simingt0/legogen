import { createRouter, createWebHistory } from "vue-router";
import LandingPage from "./pages/LandingPage.vue";
import LoadingPage from "./pages/LoadingPage.vue";
import BuildingPage from "./pages/BuildingPage.vue";

const routes = [
  {
    path: "/",
    name: "landing",
    component: LandingPage,
  },
  {
    path: "/loading",
    name: "loading",
    component: LoadingPage,
  },
  {
    path: "/game",
    name: "game",
    component: LoadingPage,
  },
  {
    path: "/build",
    name: "build",
    component: BuildingPage,
  },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
