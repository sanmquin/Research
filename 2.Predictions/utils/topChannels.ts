import { mkdirSync, writeFileSync } from "fs"
import connectDB from "../../db/connection"
import { getRecentVideos } from "./utils"
import { iVideo } from "../types"

import Channel, { ChannelDoc } from '../../db/channel'
import mongoose from "mongoose"
import Video from "../../db/video"


const getDuration = (duration:string) => {
    const match = duration.match(/PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?/);
    if (!match) return 0;
    const hours = parseInt(match[1] || "0");
    const minutes = parseInt(match[2] || "0");
    const seconds = parseInt(match[3] || "0");
    return hours * 3600 + minutes * 60 + seconds;
}

const getVideos = async (ch:ChannelDoc) => {
    const videoDocs = await Video.find({ channel: ch.name })
    const videos:iVideo[] = videoDocs.map(doc => ({
        id: doc.videoId,
        title: doc.title,
        views: doc.viewCount,
        likes: doc.likeCount,
        comments: doc.commentCount,
        date: doc.publishedAt.toISOString(),
        duration: getDuration(doc.duration)
    }))            

    return videos
}

const CHANNELS = [
    "Stokes Twins",
    "Zhong",
    "Ben Azelart",
    "The Royalty Family",
    "MSA previously My Story Animated",
    "Maizen",
    "Jordan Matter",
    "Aphmau",
    "SSundee",
    "ItsFunneh",
    "CoryxKenshin"
]

const ROOT_DIR = '../data/channels'
const getChannels = async () => {
    console.log("Fetching channels from database...")
    await connectDB()
    console.log("Connected to database.")
    const channels = await Channel.find({}).sort({ viewCount: -1 }).limit(500)
    console.log(`Fetched ${channels.length} channels from database.`)

    const topChannels:{name:string, videos:iVideo[]}[] = []

    for (const ch of channels) {
        try {
            const videos = await getVideos(ch)
            if(videos.length > 100) console.log(`Channel ${ch.name} has ${videos.length} videos.`)
            topChannels.push({ name: ch.name, videos })
        } catch (error) {
            console.error(`Error loading videos for channel ${ch.name}:`)
        }
    }

    mongoose.connection.close()
    return topChannels
}

/*
const getVideos = async (channelName: string) => {
    try {
        const channelDir = `../${ROOT_DIR}/${channelName}/videos.json`
        const { default: videos } = await import(channelDir) as { default: iRawVideo[] }
        return formatVideos(videos)
    }  catch (error) {
        console.error(`Error loading videos for channel ${channelName}:`, error)
        throw Error
    }
} 
*/

const averageViews = (videos:iVideo[]) => {
    const recentData = videos.filter(video => new Date(video.date) < new Date('2025-01-01')).slice(0, 100);
    const totalViews = recentData.reduce((sum, video) => sum + video.views, 0)
    const average = totalViews / recentData.length
    return isNaN(average) ? 0 : average
}

/*
const getTopChannelsByViews = async (topN: number) => {
    const channelNames = await getChannels()
    const channelViews: { channelName: string; avgViews: number, videos:iTrainingVideo[] }[] = []

    for (const channelName of channelNames) {
        const videos = await getVideos(channelName)
        const avgViews = averageViews(videos)
        const recentVideos = getRecentVideos(videos).map(v => ({ ...v.video, recentViews: v.recentViews }))
        channelViews.push({ channelName, avgViews, videos:recentVideos })
    }

    console.log(`Computed average views for ${channelViews.length} channels.`)
    channelViews.forEach(({ videos }) => console.log(`Channel has ${videos.length} recent videos.`))
    const qualifyingChannels = channelViews.filter(v => v.videos.length > 100)
    console.log(`Found ${qualifyingChannels.length} channels with more than 100 videos.`)

    const topChannels = qualifyingChannels.sort((a, b) => b.avgViews > a.avgViews ? 1 : -1)
    return topChannels.slice(0, topN)
}
*/



const index = async () => {
    const channels:{name:string, videos:iVideo[]}[] = []
    await connectDB()

    for(const name of CHANNELS) {
        const channelVideos = await getVideos({ name } as ChannelDoc)
        const recentVideos = channelVideos.filter(video => new Date(video.date) < new Date('2025-01-01'))
        const trainingData = getRecentVideos(recentVideos).map(v => ({ ...v.video, recentViews: v.recentViews })).slice(0, 100)
        console.log(`Channel ${name} has ${trainingData.length} recent videos.`)

        channels.push({ name, videos: trainingData })
        const dir = `./data/channels/${name}`;
        mkdirSync(dir, { recursive: true });
        writeFileSync(`${dir}/recentVideos.json`, JSON.stringify(trainingData, null, 2));
    }


    mongoose.connection.close()
    return

    /*
    const topChannels = await getTopChannelsByViews(100)
    console.log("\nTop Channels by Average Views (last 100 videos):", topChannels.length);
    topChannels.forEach(({ channelName, avgViews, videos }, index) => {
        const lastVideo = videos[videos.length - 1];
        console.log(`${index + 1}. ${channelName}: ${formatViews(avgViews)} average views. Last video: ${lastVideo?.title} (${lastVideo?.date})`)
        mkdirSync(dir, { recursive: true });
    }) */
}

// TODO: Get recent views. Write into nested directory. Filter more than 100 videos.
index().catch(console.error);
