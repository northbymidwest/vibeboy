//! GPU rendering for GTK4 via GtkGLArea + OpenGL.
//!
//! Uses GtkGLArea for GL-accelerated frame display. GTK4 owns the window
//! surface; we render into the GLArea's framebuffer, avoiding Wayland
//! protocol conflicts.

use glow::HasContext;

/// Resolve GL function pointers via dlsym. GTK4 loads the GL library
/// (libGL/libGLESv2/OpenGL.framework) so core functions are resolvable.
pub(crate) fn gl_proc_address(name: &str) -> *const std::ffi::c_void {
    #[cfg(target_os = "macos")]
    const RTLD_DEFAULT: *mut std::ffi::c_void = -2isize as *mut std::ffi::c_void;
    #[cfg(not(target_os = "macos"))]
    const RTLD_DEFAULT: *mut std::ffi::c_void = std::ptr::null_mut();

    unsafe extern "C" {
        fn dlsym(
            handle: *mut std::ffi::c_void,
            symbol: *const std::ffi::c_char,
        ) -> *mut std::ffi::c_void;
    }
    let c_name = std::ffi::CString::new(name).unwrap();
    let ptr = unsafe { dlsym(RTLD_DEFAULT, c_name.as_ptr()) };
    if !ptr.is_null() {
        return ptr as *const _;
    }
    #[cfg(target_os = "linux")]
    {
        linux_get_proc_address(&c_name)
    }
    #[cfg(not(target_os = "linux"))]
    {
        std::ptr::null()
    }
}

/// On Linux GTK4 reaches GL through libepoxy, which dlopens libEGL/libGL
/// without RTLD_GLOBAL, so the dlsym above finds nothing. Ask the loader GTK
/// already opened instead: eglGetProcAddress (GTK's default, and Mesa's
/// returns core functions too), else glXGetProcAddressARB for a GLX context.
/// RTLD_NOLOAD only finds a library that is already loaded; it never loads one.
#[cfg(target_os = "linux")]
fn linux_get_proc_address(name: &std::ffi::CStr) -> *const std::ffi::c_void {
    const RTLD_LAZY: i32 = 0x1;
    const RTLD_NOLOAD: i32 = 0x4;
    unsafe extern "C" {
        fn dlopen(filename: *const std::ffi::c_char, flag: i32) -> *mut std::ffi::c_void;
        fn dlsym(
            handle: *mut std::ffi::c_void,
            symbol: *const std::ffi::c_char,
        ) -> *mut std::ffi::c_void;
    }
    type GetProcAddress = unsafe extern "C" fn(*const std::ffi::c_char) -> *const std::ffi::c_void;
    for (lib, loader) in [
        (c"libEGL.so.1", c"eglGetProcAddress"),
        (c"libGL.so.1", c"glXGetProcAddressARB"),
    ] {
        unsafe {
            let handle = dlopen(lib.as_ptr(), RTLD_LAZY | RTLD_NOLOAD);
            if handle.is_null() {
                continue;
            }
            let get = dlsym(handle, loader.as_ptr());
            if get.is_null() {
                continue;
            }
            let get: GetProcAddress = std::mem::transmute(get);
            let ptr = get(name.as_ptr());
            if !ptr.is_null() {
                return ptr;
            }
        }
    }
    std::ptr::null()
}

pub struct GlRenderer {
    gl: glow::Context,
    program: glow::Program,
    texture: glow::Texture,
    vao: glow::VertexArray,
    tex_w: u32,
    tex_h: u32,
    /// GtkGLArea's framebuffer, recorded by `begin_frame`. wgpu compute on the
    /// shared context binds framebuffer 0, so draws bind this first, and put
    /// wgpu's binding back afterwards: wgpu expects the context as it left it,
    /// and its next compute pass writes nothing if GTK's framebuffer is bound.
    target_fb: Option<glow::Framebuffer>,
}

/// Frame data queued by the emulator tick, consumed by the GL render signal.
pub struct PendingFrame {
    pub pixels: Vec<u32>,
    pub frame_w: u32,
    pub frame_h: u32,
    pub src_w: u32,
    pub src_h: u32,
    /// If set, the render callback runs this GPU compute filter on `pixels`
    /// instead of uploading them directly. Enables zero-copy GPU→display.
    pub gpu_filter: Option<crate::scaling::wgpu_scale::WgpuScaleFilter>,
    pub fit_w: u32,
    pub fit_h: u32,
    /// Scale factor from ScaleFilter::factor() (0 = adaptive).
    pub factor: u32,
    /// Pre-rendered GL texture from GPU compute (shared-chain rasterizer).
    pub gl_texture: Option<glow::Texture>,
}

impl Default for PendingFrame {
    fn default() -> Self {
        Self {
            pixels: Vec::new(),
            frame_w: 0,
            frame_h: 0,
            src_w: 0,
            src_h: 0,
            gpu_filter: None,
            fit_w: 0,
            fit_h: 0,
            factor: 0,
            gl_texture: None,
        }
    }
}

fn compile_shader(gl: &glow::Context, ty: u32, src: &str) -> Option<glow::Shader> {
    unsafe {
        let shader = gl.create_shader(ty).ok()?;
        gl.shader_source(shader, src);
        gl.compile_shader(shader);
        if !gl.get_shader_compile_status(shader) {
            eprintln!("Shader compile error: {}", gl.get_shader_info_log(shader));
            gl.delete_shader(shader);
            return None;
        }
        Some(shader)
    }
}

impl GlRenderer {
    /// Create a new GL renderer. Must be called with a current GL context.
    pub fn new() -> Option<Self> {
        let gl = unsafe { glow::Context::from_loader_function(|s| gl_proc_address(s)) };

        let version_str = unsafe { gl.get_parameter_string(glow::VERSION) };
        let is_gles = version_str.contains("OpenGL ES");

        let version_line = if is_gles {
            "#version 300 es\nprecision mediump float;\n"
        } else {
            "#version 330\n"
        };

        let vs_src = format!(
            "{version_line}\
             out vec2 v_uv;\n\
             void main() {{\n\
                 float x = float(gl_VertexID & 1);\n\
                 float y = float((gl_VertexID >> 1) & 1);\n\
                 v_uv = vec2(x, y);\n\
                 gl_Position = vec4(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);\n\
             }}\n"
        );

        let fs_src = format!(
            "{version_line}\
             in vec2 v_uv;\n\
             out vec4 frag_color;\n\
             uniform sampler2D u_tex;\n\
             void main() {{\n\
                 frag_color = texture(u_tex, v_uv);\n\
             }}\n"
        );

        unsafe {
            let vs = compile_shader(&gl, glow::VERTEX_SHADER, &vs_src)?;
            let fs = compile_shader(&gl, glow::FRAGMENT_SHADER, &fs_src)?;

            let program = gl.create_program().ok()?;
            gl.attach_shader(program, vs);
            gl.attach_shader(program, fs);
            gl.link_program(program);
            gl.delete_shader(vs);
            gl.delete_shader(fs);

            if !gl.get_program_link_status(program) {
                eprintln!("Program link error: {}", gl.get_program_info_log(program));
                gl.delete_program(program);
                return None;
            }

            let texture = gl.create_texture().ok()?;
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );

            let vao = gl.create_vertex_array().ok()?;

            eprintln!(
                "GPU renderer initialized (OpenGL{})",
                if is_gles { " ES" } else { "" }
            );

            Some(GlRenderer {
                gl,
                program,
                texture,
                vao,
                tex_w: 0,
                tex_h: 0,
                target_fb: None,
            })
        }
    }

    /// The GL_RENDERER string of the current context.
    pub fn renderer_name(&self) -> String {
        unsafe { self.gl.get_parameter_string(glow::RENDERER) }
    }

    /// Record the framebuffer GtkGLArea bound for this render signal. Call at
    /// the start of the signal, before anything else can change the binding.
    pub fn begin_frame(&mut self) {
        unsafe {
            let fb = self.gl.get_parameter_i32(glow::DRAW_FRAMEBUFFER_BINDING);
            self.target_fb = std::num::NonZeroU32::new(fb as u32).map(glow::NativeFramebuffer);
        }
    }

    /// Bind GtkGLArea's framebuffer for a draw, returning the previous binding.
    unsafe fn bind_target(&self) -> Option<glow::Framebuffer> {
        unsafe {
            let prev = self.gl.get_parameter_i32(glow::FRAMEBUFFER_BINDING);
            self.gl.bind_framebuffer(glow::FRAMEBUFFER, self.target_fb);
            std::num::NonZeroU32::new(prev as u32).map(glow::NativeFramebuffer)
        }
    }

    /// Restore the framebuffer binding saved by `bind_target`.
    unsafe fn restore_fb(&self, prev: Option<glow::Framebuffer>) {
        unsafe { self.gl.bind_framebuffer(glow::FRAMEBUFFER, prev) };
    }

    /// Clear the current GL framebuffer to black (no frame to show yet).
    pub fn clear(&self, viewport_w: i32, viewport_h: i32) {
        unsafe {
            let gl = &self.gl;
            let prev_fb = self.bind_target();
            gl.viewport(0, 0, viewport_w, viewport_h);
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            self.restore_fb(prev_fb);
        }
    }

    /// Render pixels to the current GL framebuffer (GtkGLArea's FBO).
    pub fn render(
        &mut self,
        pixels: &[u32],
        frame_w: u32,
        frame_h: u32,
        viewport_w: i32,
        viewport_h: i32,
        src_w: u32,
        src_h: u32,
    ) {
        let npx = frame_w as usize * frame_h as usize;
        if pixels.len() < npx {
            return;
        }

        unsafe {
            let gl = &self.gl;

            // Clear full viewport to black
            let prev_fb = self.bind_target();
            gl.viewport(0, 0, viewport_w, viewport_h);
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);

            // Aspect-ratio-correct sub-viewport
            let scale = (viewport_w as f32 / src_w as f32).min(viewport_h as f32 / src_h as f32);
            let vp_w = (src_w as f32 * scale) as i32;
            let vp_h = (src_h as f32 * scale) as i32;
            let vp_x = (viewport_w - vp_w) / 2;
            let vp_y = (viewport_h - vp_h) / 2;
            gl.viewport(vp_x, vp_y, vp_w, vp_h);

            // Convert 0x00RRGGBB -> RGBA bytes
            let rgba: Vec<u8> = pixels[..npx]
                .iter()
                .flat_map(|&p| {
                    let r = ((p >> 16) & 0xFF) as u8;
                    let g = ((p >> 8) & 0xFF) as u8;
                    let b = (p & 0xFF) as u8;
                    [r, g, b, 0xFF]
                })
                .collect();

            // Upload texture
            gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));
            if self.tex_w != frame_w || self.tex_h != frame_h {
                gl.tex_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    glow::RGBA8 as i32,
                    frame_w as i32,
                    frame_h as i32,
                    0,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(&rgba)),
                );
                self.tex_w = frame_w;
                self.tex_h = frame_h;
            } else {
                gl.tex_sub_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    0,
                    0,
                    frame_w as i32,
                    frame_h as i32,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(Some(&rgba)),
                );
            }

            // Draw fullscreen quad
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vao));
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            self.restore_fb(prev_fb);
        }
    }

    /// Blit an existing GL texture (e.g., from wgpu compute output) to the framebuffer.
    /// Zero-copy path — no pixel upload needed.
    pub fn render_gl_texture(
        &self,
        texture: glow::Texture,
        viewport_w: i32,
        viewport_h: i32,
        src_w: u32,
        src_h: u32,
    ) {
        unsafe {
            let gl = &self.gl;

            // Clear full viewport to black
            let prev_fb = self.bind_target();
            gl.viewport(0, 0, viewport_w, viewport_h);
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);

            // Aspect-ratio-correct sub-viewport
            let scale = (viewport_w as f32 / src_w as f32).min(viewport_h as f32 / src_h as f32);
            let vp_w = (src_w as f32 * scale) as i32;
            let vp_h = (src_h as f32 * scale) as i32;
            let vp_x = (viewport_w - vp_w) / 2;
            let vp_y = (viewport_h - vp_h) / 2;
            gl.viewport(vp_x, vp_y, vp_w, vp_h);

            // Bind the external texture and set nearest filtering
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );

            // Draw fullscreen quad
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vao));
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);

            // Rebind our own texture so we don't hold a reference to the external one
            gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));
            self.restore_fb(prev_fb);
        }
    }
}

impl Drop for GlRenderer {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_program(self.program);
            self.gl.delete_texture(self.texture);
            self.gl.delete_vertex_array(self.vao);
        }
    }
}
