export interface iVideo {
    id: string;
    title: string;
    views: number;
    likes: number;
    comments: number;
    date: string;
    duration: number;
}

export interface iFeature {
    name: string;
    summary: string;
    description: string;
}

export interface iLabeledFeature extends iFeature { cluster: number }


export interface iRecentViews {
    video: iVideo;
    recentViews: number[];
}[]

export interface iTrainingVideo extends iVideo {
    recentViews: number[];
}

export interface iEmbeddedVideo extends iTrainingVideo {
    embedding: number[];
}

export interface iFeaturedVideo extends iTrainingVideo { features: Record<string, number> }

export interface iPrediction {
    title: string;
    actual: number;
    predicted: number;
    diff: number;
    delta: number;
}


export interface iEmbeddings {
    embeddedVideos: iEmbeddedVideo[]
    embeddedTestingVideos: iEmbeddedVideo[]
}
